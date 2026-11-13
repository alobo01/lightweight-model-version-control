#!/usr/bin/env python3
"""Validate the experiment registry and compile it into LINEAGE.md.

Usage:
    python scripts/registry.py           # validate, then write LINEAGE.md
    python scripts/registry.py --check   # validate, then print the document to stdout
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
STATUSES = ("planned", "running", "paused", "accepted", "inconclusive")
IN_PROGRESS = ("planned", "running", "paused")
REQUIRED = ("id", "title", "phase", "owner", "status", "code", "inputs")
REJECTED_ISSUES = "https://github.com/alobo01/lightweight-model-version-control/issues?q=is%3Aissue+is%3Aclosed+label%3Arejected"


def load(root: Path = ROOT) -> tuple[dict, list[dict]]:
    pipeline = _hashes_as_text(yaml.safe_load((root / "pipeline.yaml").read_text()))
    experiments = [
        _hashes_as_text(yaml.safe_load(path.read_text()))
        for path in sorted((root / "experiments").glob("*.yaml"))
        if not path.name.startswith("_")
    ]
    return pipeline, experiments


def _hashes_as_text(node):
    """YAML reads an all-digit hash such as 0123456 as a number; keep revisions and commits as text."""
    if isinstance(node, dict):
        return {
            k: str(v) if k in ("revision", "commit") and isinstance(v, int) else _hashes_as_text(v)
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_hashes_as_text(v) for v in node]
    return node


def validate(pipeline: dict, experiments: list[dict]) -> list[str]:
    """Return a list of human-readable problems; empty means the registry is valid."""
    errors: list[str] = []
    phases = {p["id"] for p in pipeline.get("phases", [])}
    benchmarks = pipeline.get("benchmarks", [])
    models: dict[str, tuple[str, dict]] = {}  # model id -> (where it is defined, hf pointer)
    seen_experiments: set[str] = set()

    def register(where: str, model: dict) -> None:
        mid = model.get("id")
        hf = _mapping(model.get("hf"))
        if not mid:
            errors.append(f"{where}: model without an id")
            return
        if mid in models:
            errors.append(f"{where}: model id '{mid}' already defined by {models[mid][0]}")
        else:
            models[mid] = (where, hf)
        if not hf.get("repo") or not hf.get("revision"):
            errors.append(f"{where}: model '{mid}' needs hf.repo and hf.revision")
        if not _mapping(model.get("eval")).get("commit"):
            errors.append(f"{where}: model '{mid}' needs eval.commit (evaluation code version)")
        scores = _mapping(model.get("scores"))
        missing = [b for b in benchmarks if b not in scores]
        if missing:
            errors.append(f"{where}: model '{mid}' is missing scores for {', '.join(missing)}")
        for name, value in scores.items():
            if name not in benchmarks:
                errors.append(f"{where}: model '{mid}' has unknown benchmark '{name}'")
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"{where}: model '{mid}' score '{name}' is not a number")

    for base in pipeline.get("base_models", []):
        register("pipeline.yaml", base)

    for exp in experiments:
        eid = exp.get("id", "<missing id>")
        for field in REQUIRED:
            if exp.get(field) in (None, "", [], {}):
                errors.append(f"{eid}: missing '{field}'")
        if eid in seen_experiments:
            errors.append(f"{eid}: duplicate experiment id")
        seen_experiments.add(eid)
        if exp.get("phase") not in phases:
            errors.append(f"{eid}: unknown phase '{exp.get('phase')}'")
        if exp.get("status") not in STATUSES:
            errors.append(f"{eid}: unknown status '{exp.get('status')}' (rejected work goes to an issue)")
        owner = exp.get("owner") or {}
        if not isinstance(owner, dict) or not all(owner.get(k) for k in ("name", "email", "github")):
            errors.append(f"{eid}: owner needs name, email and github")
        elif "@" not in owner["email"]:
            errors.append(f"{eid}: owner email '{owner['email']}' is not valid")
        code = _mapping(exp.get("code"))
        if exp.get("code") and not (code.get("repo") and code.get("commit")):
            errors.append(f"{eid}: code needs repo and commit")
        elif exp.get("code") and not re.fullmatch(r"[\w.-]+/[\w.-]+", str(code["repo"])):
            errors.append(f"{eid}: code.repo '{code['repo']}' must be a GitHub repository as owner/name")
        for out in exp.get("outputs") or []:
            register(eid, out)

    for exp in experiments:
        eid = exp.get("id")
        inputs = exp.get("inputs") or []
        for inp in inputs:
            mid = inp.get("model")
            if mid not in models:
                errors.append(f"{eid}: input model '{mid}' is not registered")
                continue
            hf, registered = _mapping(inp.get("hf")), models[mid][1]
            if not hf.get("repo") or not hf.get("revision"):
                errors.append(f"{eid}: input '{mid}' needs hf.repo and hf.revision")
            elif (hf["repo"], hf["revision"]) != (registered.get("repo"), registered.get("revision")):
                errors.append(
                    f"{eid}: input '{mid}' points to {hf['repo']}@{hf['revision']}, "
                    f"but it is registered as {registered.get('repo')}@{registered.get('revision')}"
                )
        if exp.get("phase") == "souping":
            weights = [inp.get("weight") for inp in inputs]
            if len(inputs) < 2:
                errors.append(f"{eid}: a soup needs at least two input checkpoints")
            elif any(not isinstance(w, (int, float)) for w in weights):
                errors.append(f"{eid}: every soup input needs a numeric 'weight'")
            elif abs(sum(weights) - 1) > 1e-6:
                errors.append(f"{eid}: soup weights sum to {sum(weights):g}, not 1")

    main = pipeline.get("main")
    if main not in models:
        errors.append(f"pipeline.yaml: main '{main}' is not a registered model")

    if not errors and _has_cycle(experiments):
        errors.append("lineage contains a cycle")
    return errors


def _mapping(value) -> dict:
    """Treat anything that is not a mapping as missing, so validation reports it."""
    return value if isinstance(value, dict) else {}


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
    classes: dict[str, list[str]] = {"lineage": [], "progress": [], "main": []}

    def classify(identifier: str, status: str | None) -> None:
        if identifier == pipeline["main"]:
            classes["main"].append(_node(identifier))
        elif status in IN_PROGRESS:
            classes["progress"].append(_node(identifier))
        elif identifier in lineage:
            classes["lineage"].append(_node(identifier))

    def model_node(model: dict) -> str:
        suffix = " · main" if model["id"] == pipeline["main"] else ""
        return f'  {_node(model["id"])}(["{model["id"]}{suffix}<br/>@{model["hf"]["revision"]}"])'

    for base in pipeline.get("base_models", []):
        lines.append(model_node(base))
        classify(base["id"], None)

    for exp in experiments:
        eid, status = exp["id"], exp["status"]
        code = f'{exp["code"]["repo"]}@{exp["code"]["commit"]}'
        lines.append(f'  {_node(eid)}["{eid} · {phase_label[exp["phase"]]}<br/>{status}<br/><small>{code}</small>"]')
        classify(eid, status)
        for inp in exp["inputs"]:
            label = f'|{inp["weight"]}|' if "weight" in inp else ""
            lines.append(f'  {_node(inp["model"])} -->{label} {_node(eid)}')
        for out in exp.get("outputs") or []:
            lines.append(model_node(out))
            lines.append(f'  {_node(eid)} --> {_node(out["id"])}')
            classify(out["id"], status)

    lines += [
        "  classDef lineage fill:#dbeafe,stroke:#2563eb,color:#1e3a8a",
        "  classDef main fill:#dcfce7,stroke:#15803d,color:#14532d,stroke-width:3px",
        "  classDef progress fill:#fef9c3,stroke:#a16207,color:#713f12,stroke-dasharray:2 3",
    ]
    lines += [f"  class {','.join(ids)} {name}" for name, ids in classes.items() if ids]
    return "\n".join(lines)


def render(pipeline: dict, experiments: list[dict]) -> str:
    phase_order = {p["id"]: n for n, p in enumerate(pipeline["phases"])}
    phase_label = {p["id"]: p["label"] for p in pipeline["phases"]}
    benchmarks = pipeline.get("benchmarks", [])
    ordered = sorted(experiments, key=lambda e: (phase_order[e["phase"]], e["id"]))
    used_by: dict[str, list[str]] = {}
    for exp in experiments:
        for inp in exp["inputs"]:
            used_by.setdefault(inp["model"], []).append(exp["id"])

    def hf_link(hf: dict) -> str:
        url = f'https://huggingface.co/{hf["repo"]}/tree/{hf["revision"]}'
        return f'[{hf["repo"]}@{hf["revision"]}]({url})'

    def commit_link(repo: str, commit: str) -> str:
        return f"[{repo}@{commit}](https://github.com/{repo}/commit/{commit})"

    def code_link(code: dict) -> str:
        repo, commit = code["repo"], code["commit"]
        return (f"[{repo}](https://github.com/{repo}) @ "
                f"[{commit}](https://github.com/{repo}/commit/{commit})")

    def owner(exp: dict) -> str:
        o = exp["owner"]
        return f'[{o["name"]}](mailto:{o["email"]}) (`{o["github"]}`)'

    def inputs(exp: dict) -> str:
        return "<br/>".join(
            f'`{i["model"]}` {hf_link(i["hf"])}' + (f' ×{i["weight"]}' if "weight" in i else "")
            for i in exp["inputs"]
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
        "Green: current main · blue: ancestry of main · yellow: in progress · "
        "unshaded: accepted, not in main's ancestry.",
        "",
        "## Experiments",
        "",
        "| Experiment | Phase | Objective | Owner | Status | Parents | Outputs | GitHub code |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for exp in ordered:
        outs = ", ".join(f'`{o["id"]}`' for o in exp.get("outputs") or []) or "–"
        doc.append(
            f'| {exp["id"]} | {phase_label[exp["phase"]]} | {exp["title"]} | {owner(exp)} | '
            f'{exp["status"]} | {inputs(exp)} | {outs} | '
            f'{code_link(exp["code"])} |'
        )

    doc += [
        "",
        "## Checkpoints",
        "",
        f"Scores produced by [`{pipeline['eval_repo']}`](https://github.com/{pipeline['eval_repo']}) "
        "at the listed commit.",
        "",
        "| Model | Produced by | Checkpoint | " + " | ".join(benchmarks) + " | Eval code | Used by |",
        "| --- | --- | --- | " + " | ".join("---:" for _ in benchmarks) + " | --- | --- |",
    ]

    def model_row(model: dict, source: str) -> str:
        name = f'**`{model["id"]}`** (main)' if model["id"] == pipeline["main"] else f'`{model["id"]}`'
        scores = " | ".join(f'{model["scores"][b]:g}' for b in benchmarks)
        commit = model["eval"]["commit"]
        eval_link = f"[{commit}](https://github.com/{pipeline['eval_repo']}/commit/{commit})"
        users = ", ".join(used_by.get(model["id"], [])) or "–"
        return f'| {name} | {source} | {hf_link(model["hf"])} | {scores} | {eval_link} | {users} |'

    doc += [model_row(base, "external") for base in pipeline.get("base_models", [])]
    doc += [model_row(out, exp["id"]) for exp in ordered for out in exp.get("outputs") or []]

    doc += [
        "",
        "## Rejected experiments",
        "",
        f"Rejected work is kept as [closed issues labelled `rejected`]({REJECTED_ISSUES}), "
        "with its parents, checkpoints, scores and the reason it was dropped.",
    ]
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
