#!/usr/bin/env python3
"""Validate the experiment registry and compile it into LINEAGE.md.

Usage:
    python scripts/registry.py           # validate, then write LINEAGE.md
    python scripts/registry.py --check   # validate, then print the document to stdout
"""
from __future__ import annotations

import colorsys
import hashlib
import html
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
STATUSES = ("in_progress", "accepted", "rejected")
REQUIRED = ("id", "title", "owner", "status", "code", "inputs")


def load(root: Path = ROOT) -> tuple[dict, list[dict]]:
    config = yaml.safe_load((root / "config.yaml").read_text())
    return config, _load_experiments(root / "experiments")


def load_rejected(root: Path = ROOT) -> list[dict]:
    return _load_experiments(root / "rejected")


def _load_experiments(directory: Path) -> list[dict]:
    records = []
    for path in sorted(directory.glob("*.yaml")):
        if not path.name.startswith("_"):
            record = _hashes_as_text(yaml.safe_load(path.read_text()))
            record["_source"] = f"{directory.name}/{path.name}"
            records.append(record)
    return records


def _hashes_as_text(node):
    """YAML reads an all-digit hash such as 0123456 as a number; keep revisions and commits as text."""
    if isinstance(node, dict):
        return {
            k: v if k in ("parameters", "config") else (
                str(v) if k in ("revision", "commit") and isinstance(v, int) else _hashes_as_text(v)
            )
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_hashes_as_text(v) for v in node]
    return node


def validate(config: dict, experiments: list[dict], rejected: list[dict] | None = None) -> list[str]:
    """Return a list of human-readable problems; empty means the registry is valid."""
    errors: list[str] = []
    if not isinstance(config, dict):
        return ["config.yaml: needs project_name and main_branch"]
    for field in ("project_name", "main_branch"):
        value = config.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"config.yaml: {field} must be a nonempty string")
    for field in config:
        if field not in ("project_name", "main_branch"):
            errors.append(f"config.yaml: unknown field '{field}'")
    branch = config.get("main_branch")
    if isinstance(branch, str) and branch.strip():
        if "\0" in branch or subprocess.run(
            ["git", "check-ref-format", f"refs/heads/{branch}"], capture_output=True
        ).returncode:
            errors.append("config.yaml: main_branch must be a valid Git branch name")
    active = experiments
    rejected = rejected or []
    for exp in active:
        if exp.get("status") == "rejected":
            errors.append(f"{exp.get('id')}: move rejected experiments to rejected/")
    for exp in rejected:
        if exp.get("status") != "rejected":
            errors.append(f"{exp.get('id')}: files in rejected/ need status: rejected")
    rejected_models = {out["id"] for exp in rejected for out in exp.get("outputs") or []}
    for exp in active:
        for inp in exp.get("inputs") or []:
            if inp.get("model") in rejected_models:
                errors.append(f"{exp.get('id')}: input '{inp['model']}' belongs to a rejected experiment")
    experiments = active + rejected
    models: dict[str, tuple[str, dict]] = {}  # model id -> (where it is defined, hf pointer)
    seen_experiments: set[str] = set()

    def register(where: str, model: dict, external: bool = False) -> None:
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
        if external and "eval" not in model and "scores" not in model:
            return
        evaluation = _mapping(model.get("eval"))
        if not evaluation.get("repo"):
            errors.append(f"{where}: model '{mid}' needs eval.repo (evaluation code repository)")
        elif not re.fullmatch(r"[\w.-]+/[\w.-]+", str(evaluation["repo"])):
            errors.append(f"{where}: model '{mid}' eval.repo must be a GitHub repository as owner/name")
        if not evaluation.get("commit"):
            errors.append(f"{where}: model '{mid}' needs eval.commit (evaluation code version)")
        scores = _mapping(model.get("scores"))
        if not scores:
            errors.append(f"{where}: model '{mid}' needs nonempty scores")
        for name, value in scores.items():
            if not isinstance(name, str) or not name.strip():
                errors.append(f"{where}: model '{mid}' benchmark names must be nonempty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"{where}: model '{mid}' score '{name}' is not a number")

    for base in external_models(experiments):
        register("external input", base, external=True)

    for exp in experiments:
        eid = exp.get("id", "<missing id>")
        for field in REQUIRED:
            if exp.get(field) in (None, "", [], {}):
                errors.append(f"{eid}: missing '{field}'")
        if eid in seen_experiments:
            errors.append(f"{eid}: duplicate experiment id")
        seen_experiments.add(eid)
        tags = exp.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag.strip() for tag in tags):
            errors.append(f"{eid}: tags must be a list of nonempty strings")
        if exp.get("status") not in STATUSES:
            errors.append(f"{eid}: unknown status '{exp.get('status')}'")
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
        method = exp.get("method")
        if not isinstance(method, str) or not method.strip():
            errors.append(f"{eid}: needs a nonempty string 'method'")
        if "config" in exp:
            run_config = exp["config"]
            if not isinstance(run_config, dict) or not any(k in run_config for k in ("file", "cli")):
                errors.append(f"{eid}: config needs a 'file' or 'cli' field")
            else:
                for key, value in run_config.items():
                    if key not in ("file", "cli"):
                        errors.append(f"{eid}: unknown config field '{key}'")
                    elif not isinstance(value, str) or not value.strip():
                        errors.append(f"{eid}: config.{key} must be a nonempty string")
        for out in exp.get("outputs") or []:
            register(eid, out)

    for exp in experiments:
        eid = exp.get("id")
        inputs = exp.get("inputs") or []
        for inp in inputs:
            mid = inp.get("model")
            if "parameters" in inp and not isinstance(inp["parameters"], dict):
                errors.append(f"{eid}: input '{mid}' parameters must be a dictionary")
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


def external_models(experiments: list[dict]) -> list[dict]:
    """Inputs with no producing experiment are external checkpoints."""
    produced = {out.get("id") for exp in experiments for out in exp.get("outputs") or []}
    external: dict[str, dict] = {}
    for exp in experiments:
        for inp in exp.get("inputs") or []:
            mid = inp.get("model")
            if mid and mid not in produced:
                model = external.setdefault(mid, {"id": mid, "hf": inp.get("hf")})
                for key in ("eval", "scores"):
                    if key in inp:
                        model.setdefault(key, inp[key])
    return [external[mid] for mid in sorted(external)]


def _node(identifier: str) -> str:
    return identifier.replace("-", "_")


def _metadata(mapping: dict) -> str:
    """Keep metadata in declaration order, including nested YAML values."""
    return ", ".join(
        f"{key}: {value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)}"
        for key, value in mapping.items()
    )


def _table_text(value: str) -> str:
    value = html.escape(value, quote=False).replace("|", "&#124;").replace("`", "&#96;")
    return value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br/>")


def _mermaid_text(value: str) -> str:
    # Mermaid uses #number; entities rather than HTML's &#number; syntax.
    return "".join(f"#{ord(c)};" if c in '#&"<>|[]{}\\`\r\n' else c for c in value)


def _tag_styles(experiments: list[dict]) -> dict[str, tuple[str, str, str]]:
    """Generate a distinct class and colours for each discovered tag, without configuration."""
    tags = sorted({tag for exp in experiments for tag in exp.get("tags", [])})
    styles = {}
    used = set()
    for index, tag in enumerate(tags):
        # Derive the colour from the tag so adding tags usually leaves existing colours unchanged.
        hue = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "big") / 2**32

        def colour(lightness: float) -> str:
            rgb = colorsys.hls_to_rgb(hue, lightness, 0.65)
            return "#" + "".join(f"{round(c * 255):02x}" for c in rgb)

        fill = colour(0.86)
        while fill in used:
            hue = (hue + 0.001) % 1
            fill = colour(0.86)
        used.add(fill)
        styles[tag] = (f"tag_{index}", fill, colour(0.38))
    return styles


def render_mermaid(config: dict, experiments: list[dict], rejected: list[dict] | None = None) -> str:
    experiments = experiments + (rejected or [])
    tag_styles = _tag_styles(experiments)
    identifiers = {_node(e["id"]) for e in experiments}
    identifiers.update(_node(o["id"]) for e in experiments for o in e.get("outputs") or [])
    identifiers.update(_node(b["id"]) for b in external_models(experiments))
    canvas = "lineage_canvas"
    while canvas in identifiers:
        canvas = "_" + canvas
    lines = [
        "---",
        "config:",
        "  theme: base",
        "  fontSize: 24",
        '  themeCSS: "& { background: #f8fafc; border-radius: 16px; }"',
        "  themeVariables:",
        '    fontSize: "24px"',
        '    fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif"',
        '    background: "#f8fafc"',
        '    primaryColor: "#ffffff"',
        '    primaryTextColor: "#1e293b"',
        '    primaryBorderColor: "#cbd5e1"',
        '    lineColor: "#64748b"',
        '    edgeLabelBackground: "#f8fafc"',
        '    clusterBkg: "#f1f5f9"',
        '    clusterBorder: "#e2e8f0"',
        "  flowchart:",
        "    useMaxWidth: false",
        "    nodeSpacing: 70",
        "    rankSpacing: 90",
        "    diagramPadding: 24",
        "    curve: basis",
        "---",
        "flowchart LR",
        f'  subgraph {canvas}["Experiment lineage"]',
        "    direction LR",
    ]
    classes: dict[str, list[str]] = {}
    edges: dict[str, list[int]] = {status: [] for status in STATUSES}
    edge_index = 0

    def edge(source: str, target: str, status: str, label: str = "") -> None:
        nonlocal edge_index
        lines.append(f'  {_node(source)} -->{label} {_node(target)}')
        edges[status].append(edge_index)
        edge_index += 1

    def classify(identifier: str, tags: list[str]) -> None:
        if tags:
            classes.setdefault(tag_styles[tags[0]][0], []).append(_node(identifier))

    def model_node(model: dict) -> str:
        return f'  {_node(model["id"])}(["{model["id"]}<br/>@{model["hf"]["revision"]}"])'

    for base in external_models(experiments):
        lines.append(model_node(base))
        classify(base["id"], [])

    for exp in experiments:
        eid, status = exp["id"], exp["status"]
        code = f'{exp["code"]["repo"]}@{exp["code"]["commit"]}'
        method = f'<br/>{_mermaid_text(str(exp["method"]))}' if exp.get("method") else ""
        tags = exp.get("tags", [])
        tag_label = f'<br/><small>{_mermaid_text(", ".join(tags))}</small>' if tags else ""
        lines.append(f'  {_node(eid)}("<b>{eid}</b>{tag_label}{method}<br/>{status.replace("_", " ")}<br/><small>{code}</small>")')
        classify(eid, tags)
        for inp in exp["inputs"]:
            parameters = _metadata(inp.get("parameters", {}))
            label = f'|"{_mermaid_text(parameters)}"|' if parameters else ""
            edge(inp["model"], eid, status, label)
        for out in exp.get("outputs") or []:
            lines.append(model_node(out))
            edge(eid, out["id"], status)
            classify(out["id"], tags)

    if tag_styles:
        # Prefix legend IDs until they cannot collide with real experiment/model IDs.
        prefix = "tag_legend"
        while prefix == canvas or prefix in identifiers or any(f"{prefix}_{i}" in identifiers for i in range(len(tag_styles))):
            prefix = "_" + prefix
        lines.append(f'  subgraph {prefix}["Tags"]')
        for index, (tag, (name, _, _)) in enumerate(tag_styles.items()):
            legend_id = f"{prefix}_{index}"
            lines.append(f'    {legend_id}["{_mermaid_text(tag)}"]')
            classes.setdefault(name, []).append(legend_id)
        lines.append("  end")

    lines += ["  end", f"  style {canvas} fill:#f8fafc,stroke:#e2e8f0,stroke-width:1px,rx:16,ry:16"]
    lines.append("  classDef default fill:#ffffff,stroke:#cbd5e1,color:#1e293b,stroke-width:1.5px")
    lines += [f"  classDef {name} fill:{fill},stroke:{stroke},color:#1e293b,stroke-width:1.5px"
              for name, fill, stroke in tag_styles.values()]
    lines += [f"  class {','.join(ids)} {name}" for name, ids in classes.items() if ids]
    edge_styles = {
        "accepted": "stroke:#16a34a,stroke-width:3px",
        "rejected": "stroke:#dc2626,stroke-width:3px",
        "in_progress": "stroke:#64748b,stroke-width:3px,stroke-dasharray:8 6",
    }
    for status, indices in edges.items():
        if indices:
            lines.append(f"  linkStyle {','.join(map(str, indices))} {edge_styles[status]}")
    return "\n".join(lines)


def render(config: dict, experiments: list[dict], rejected: list[dict] | None = None) -> str:
    bases = external_models(experiments)
    checkpoints = bases + [out for exp in experiments for out in exp.get("outputs") or []]
    benchmarks = sorted({name for model in checkpoints for name in model.get("scores", {})})
    ordered = sorted(experiments, key=lambda e: e["id"])
    used_by: dict[str, list[str]] = {}
    for exp in experiments:
        for inp in exp["inputs"]:
            used_by.setdefault(inp["model"], []).append(exp["id"])

    def hf_link(hf: dict) -> str:
        url = f'https://huggingface.co/{hf["repo"]}/tree/{hf["revision"]}'
        return f'[{hf["repo"]}@{hf["revision"]}]({url})'

    def code_link(code: dict) -> str:
        repo, commit = code["repo"], code["commit"]
        return (f"[{repo}](https://github.com/{repo}) @ "
                f"[{commit}](https://github.com/{repo}/commit/{commit})")

    def owner(exp: dict) -> str:
        o = exp["owner"]
        return f'[{o["name"]}](mailto:{o["email"]}) (`{o["github"]}`)'

    def inputs(exp: dict) -> str:
        return "<br/>".join(
            f'`{i["model"]}` {hf_link(i["hf"])}'
            + (f' ({_table_text(_metadata(i["parameters"]))})' if i.get("parameters") else "")
            for i in exp["inputs"]
        )

    doc = [
        f"# {config['project_name']} — lineage",
        "",
        "> Generated by `scripts/registry.py` from `config.yaml`, `experiments/` and `rejected/`. Do not edit by hand.",
        "",
        f"**Main branch:** `{config['main_branch']}`",
        "",
        "```mermaid",
        render_mermaid(config, experiments, rejected),
        "```",
        "",
        "Node colours identify tags (see legend); the first tag colours each experiment and its outputs. "
        "Arrows show experiment status: green = accepted · red = rejected · dashed grey = in progress. "
        "Both input and output arrows use the status of their experiment.",
        "",
        "## Experiments",
        "",
        "| Experiment | Tags | Method | Objective | Owner | Status | Parents | Outputs | GitHub code | Configuration |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for exp in ordered:
        outs = ", ".join(f'`{o["id"]}`' for o in exp.get("outputs") or []) or "–"
        doc.append(
            f'| {exp["id"]} | {_table_text(", ".join(exp.get("tags", []))) or "–"} | {_table_text(str(exp.get("method", "–")))} | {exp["title"]} | {owner(exp)} | '
            f'{exp["status"].replace("_", " ")} | {inputs(exp)} | {outs} | '
            f'{code_link(exp["code"])} | {_table_text(_metadata(exp["config"])) if exp.get("config") else "–"} |'
        )

    doc += [
        "",
        "## Checkpoints",
        "",
        "Benchmark columns come from recorded scores; missing scores are shown as –. "
        "Scores were produced by the evaluation code listed for each checkpoint.",
        "",
        "| " + " | ".join(["Model", "Produced by", "Checkpoint", *(_table_text(b) for b in benchmarks), "Eval code", "Used by"]) + " |",
        "| " + " | ".join(["---", "---", "---", *("---:" for _ in benchmarks), "---", "---"]) + " |",
    ]

    def model_row(model: dict, source: str) -> str:
        name = f'`{model["id"]}`'
        scores = [f'{model["scores"][b]:g}' if b in model.get("scores", {}) else "–" for b in benchmarks]
        evaluation = model.get("eval", {})
        eval_link = code_link(evaluation) if evaluation else "–"
        users = ", ".join(used_by.get(model["id"], [])) or "–"
        return "| " + " | ".join([name, source, hf_link(model["hf"]), *scores, eval_link, users]) + " |"

    doc += [model_row(base, "external") for base in bases]
    doc += [model_row(out, exp["id"]) for exp in ordered for out in exp.get("outputs") or []]

    doc += [
        "",
        "## Rejected experiments",
        "",
        "Rejected experiment records are kept in [rejected/](rejected/) with `status: rejected` "
        "and shown as red branches in the overview. Their checkpoints are excluded from the active tables above.",
    ]
    if rejected:
        doc += ["", "| Experiment | Objective | Reason |", "| --- | --- | --- |"]
        for exp in sorted(rejected, key=lambda e: e["id"]):
            doc.append(
                f'| [{_table_text(exp["id"])}]({exp["_source"]}) | '
                f'{_table_text(exp["title"])} | {_table_text(str(exp.get("reject_reason", "–")))} |'
            )
    else:
        doc += ["", "No rejected experiments recorded."]
    return "\n".join(doc) + "\n"


def main(argv: list[str]) -> int:
    config, experiments = load()
    rejected = load_rejected()
    errors = validate(config, experiments, rejected)
    if errors:
        print("Registry is invalid:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    document = render(config, experiments, rejected)
    if "--check" in argv:
        print(document)
    else:
        (ROOT / "LINEAGE.md").write_text(document)
        print(f"Wrote LINEAGE.md ({len(experiments)} experiments, {len(rejected)} rejected)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
