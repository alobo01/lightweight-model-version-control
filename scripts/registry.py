#!/usr/bin/env python3
"""Validate the experiment registry and compile it into LINEAGE.md.

Usage:
    python scripts/registry.py           # validate, then write LINEAGE.md
    python scripts/registry.py --check   # validate, then print the document to stdout
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
STATUSES = ("planned", "running", "paused", "accepted", "discarded", "inconclusive")
IN_PROGRESS = ("planned", "running", "paused")
REQUIRED = ("id", "title", "phase", "owner", "status", "inputs")


def load(root: Path = ROOT) -> tuple[dict, list[dict]]:
    pipeline = yaml.safe_load((root / "pipeline.yaml").read_text())
    experiments = [
        yaml.safe_load(path.read_text())
        for path in sorted((root / "experiments").glob("*.yaml"))
        if not path.name.startswith("_")
    ]
    return pipeline, experiments


def validate(pipeline: dict, experiments: list[dict]) -> list[str]:
    """Return a list of human-readable problems; empty means the registry is valid."""
    errors: list[str] = []
    phases = {p["id"] for p in pipeline.get("phases", [])}
    models: dict[str, str] = {}  # model id -> where it is defined
    seen_experiments: set[str] = set()

    def check_hf(owner: str, model: dict) -> None:
        hf = model.get("hf") or {}
        if not hf.get("repo") or not hf.get("revision"):
            errors.append(f"{owner}: model '{model.get('id')}' needs hf.repo and hf.revision")

    def register(owner: str, model: dict) -> None:
        mid = model.get("id")
        if not mid:
            errors.append(f"{owner}: model without an id")
        elif mid in models:
            errors.append(f"{owner}: model id '{mid}' already defined by {models[mid]}")
        else:
            models[mid] = owner
        check_hf(owner, model)

    for base in pipeline.get("base_models", []):
        register("pipeline.yaml", base)

    for exp in experiments:
        eid = exp.get("id", "<missing id>")
        for field in REQUIRED:
            if exp.get(field) in (None, "", []):
                errors.append(f"{eid}: missing '{field}'")
        if eid in seen_experiments:
            errors.append(f"{eid}: duplicate experiment id")
        seen_experiments.add(eid)
        if exp.get("phase") not in phases:
            errors.append(f"{eid}: unknown phase '{exp.get('phase')}'")
        if exp.get("status") not in STATUSES:
            errors.append(f"{eid}: unknown status '{exp.get('status')}'")
        if exp.get("status") == "discarded" and not exp.get("reason"):
            errors.append(f"{eid}: discarded experiments need a 'reason'")
        for out in exp.get("outputs") or []:
            register(eid, out)

    for exp in experiments:
        for inp in exp.get("inputs") or []:
            if inp.get("model") not in models:
                errors.append(f"{exp.get('id')}: input model '{inp.get('model')}' is not registered")

    main = pipeline.get("main")
    if main not in models:
        errors.append(f"pipeline.yaml: main '{main}' is not a registered model")

    if not errors and _has_cycle(experiments):
        errors.append("lineage contains a cycle")
    return errors


def _has_cycle(experiments: list[dict]) -> bool:
    """Kahn's algorithm over experiments: A precedes B if B consumes an output of A."""
    producer = {o["id"]: e["id"] for e in experiments for o in e.get("outputs") or []}
    deps = {
        e["id"]: {producer[i["model"]] for i in e["inputs"] if i["model"] in producer}
        for e in experiments
    }
    while deps:
        ready = [eid for eid, d in deps.items() if not d]
        if not ready:
            return True
        for eid in ready:
            del deps[eid]
        for d in deps.values():
            d.difference_update(ready)
    return False


def main_lineage(pipeline: dict, experiments: list[dict]) -> set[str]:
    """Ids of every model and experiment that the main checkpoint descends from."""
    producer = {o["id"]: e for e in experiments for o in e.get("outputs") or []}
    seen: set[str] = set()
    stack = [pipeline["main"]]
    while stack:
        mid = stack.pop()
        if mid in seen:
            continue
        seen.add(mid)
        exp = producer.get(mid)
        if exp:
            seen.add(exp["id"])
            stack.extend(i["model"] for i in exp["inputs"])
    return seen


def _node(identifier: str) -> str:
    return identifier.replace("-", "_")


def render_mermaid(pipeline: dict, experiments: list[dict]) -> str:
    lineage = main_lineage(pipeline, experiments)
    phase_label = {p["id"]: p["label"] for p in pipeline["phases"]}
    lines = ["flowchart LR"]
    classes: dict[str, list[str]] = {"lineage": [], "discarded": [], "progress": [], "main": []}

    def classify(identifier: str, status: str | None) -> None:
        if identifier == pipeline["main"]:
            classes["main"].append(_node(identifier))
        elif status == "discarded":
            classes["discarded"].append(_node(identifier))
        elif status in IN_PROGRESS:
            classes["progress"].append(_node(identifier))
        elif identifier in lineage:
            classes["lineage"].append(_node(identifier))

    for base in pipeline.get("base_models", []):
        lines.append(f'  {_node(base["id"])}(["{base["id"]}"])')
        classify(base["id"], None)

    for exp in experiments:
        eid, status = exp["id"], exp["status"]
        lines.append(f'  {_node(eid)}["{eid} · {phase_label[exp["phase"]]}<br/>{status}"]')
        classify(eid, status)
        for inp in exp["inputs"]:
            label = f'|{inp["weight"]}|' if "weight" in inp else ""
            lines.append(f'  {_node(inp["model"])} -->{label} {_node(eid)}')
        for out in exp.get("outputs") or []:
            suffix = " · main" if out["id"] == pipeline["main"] else ""
            lines.append(f'  {_node(out["id"])}(["{out["id"]}{suffix}"])')
            lines.append(f'  {_node(eid)} --> {_node(out["id"])}')
            classify(out["id"], status)

    lines += [
        "  classDef lineage fill:#dbeafe,stroke:#2563eb,color:#1e3a8a",
        "  classDef main fill:#dcfce7,stroke:#15803d,color:#14532d,stroke-width:3px",
        "  classDef discarded fill:#fee2e2,stroke:#b91c1c,color:#7f1d1d,stroke-dasharray:5 4",
        "  classDef progress fill:#fef9c3,stroke:#a16207,color:#713f12,stroke-dasharray:2 3",
    ]
    lines += [f"  class {','.join(ids)} {name}" for name, ids in classes.items() if ids]
    return "\n".join(lines)


def render(pipeline: dict, experiments: list[dict]) -> str:
    phase_order = {p["id"]: n for n, p in enumerate(pipeline["phases"])}
    phase_label = {p["id"]: p["label"] for p in pipeline["phases"]}
    ordered = sorted(experiments, key=lambda e: (phase_order[e["phase"]], e["id"]))
    used_by: dict[str, list[str]] = {}
    for exp in experiments:
        for inp in exp["inputs"]:
            used_by.setdefault(inp["model"], []).append(exp["id"])

    def hf_link(hf: dict) -> str:
        url = f'https://huggingface.co/{hf["repo"]}/tree/{hf["revision"]}'
        return f'[{hf["repo"]}@{hf["revision"]}]({url})'

    def code_link(exp: dict) -> str:
        code = exp.get("code")
        if not code:
            return "–"
        return f'[{code["repo"]}@{code["commit"]}](https://github.com/{code["repo"]}/commit/{code["commit"]})'

    def eval_link(out: dict) -> str:
        if not out.get("eval"):
            return "–"
        return f'[{out["eval"]}](https://github.com/{pipeline["eval_repo"]}/tree/main/{out["eval"]})'

    def inputs(exp: dict) -> str:
        return ", ".join(
            f'`{i["model"]}`' + (f' ×{i["weight"]}' if "weight" in i else "") for i in exp["inputs"]
        )

    doc = [
        f"# {pipeline['name']} — lineage",
        "",
        "> Generated by `scripts/registry.py` from `pipeline.yaml` and `experiments/`. Do not edit by hand.",
        "",
        f"**Current main:** `{pipeline['main']}`",
        "",
        "```mermaid",
        render_mermaid(pipeline, experiments),
        "```",
        "",
        "Green: current main · blue: ancestry of main · yellow: in progress · red: discarded · "
        "unshaded: accepted, not in main's ancestry.",
        "",
        "## Experiments",
        "",
        "| Experiment | Phase | Objective | Owner | Status | Inputs | Outputs | Code |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for exp in ordered:
        outs = ", ".join(f'`{o["id"]}`' for o in exp.get("outputs") or []) or "–"
        doc.append(
            f'| {exp["id"]} | {phase_label[exp["phase"]]} | {exp["title"]} | `{exp["owner"]}` | '
            f'{exp["status"]} | {inputs(exp)} | {outs} | {code_link(exp)} |'
        )

    doc += [
        "",
        "## Checkpoints",
        "",
        "| Model | Produced by | Checkpoint | Evaluation | Used by |",
        "| --- | --- | --- | --- | --- |",
    ]

    def model_row(model: dict, source: str) -> str:
        name = f'**`{model["id"]}`** (main)' if model["id"] == pipeline["main"] else f'`{model["id"]}`'
        users = ", ".join(used_by.get(model["id"], [])) or "–"
        return f'| {name} | {source} | {hf_link(model["hf"])} | {eval_link(model)} | {users} |'

    doc += [model_row(base, "external") for base in pipeline.get("base_models", [])]
    doc += [model_row(out, exp["id"]) for exp in ordered for out in exp.get("outputs") or []]

    discarded = [e for e in ordered if e["status"] == "discarded"]
    if discarded:
        doc += ["", "## Discarded work", ""]
        doc += [f'- **{e["id"]} — {e["title"]}** (`{e["owner"]}`): {e["reason"]}' for e in discarded]
    return "\n".join(doc) + "\n"


def main(argv: list[str]) -> int:
    pipeline, experiments = load()
    errors = validate(pipeline, experiments)
    if errors:
        print("Registry is invalid:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    document = render(pipeline, experiments)
    if "--check" in argv:
        print(document)
    else:
        (ROOT / "LINEAGE.md").write_text(document)
        print(f"Wrote LINEAGE.md ({len(experiments)} experiments)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
