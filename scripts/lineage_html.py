"""Render a self-contained, offline HTML view of the validated experiment registry."""
from __future__ import annotations

import html
import re
import textwrap
from collections import defaultdict, deque
from pathlib import Path


COLOURS = {"accepted": "#168654", "rejected": "#d34a4a", "in_progress": "#64748b"}


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def status_badge(status: str) -> str:
    return f'<span class="badge {status}">{status.replace("_", " ")}</span>'


def hf_link(model: dict) -> str:
    hf = model["hf"]
    return f'<a href="https://huggingface.co/{esc(hf["repo"])}/tree/{esc(hf["revision"])}">{esc(model.get("model", model.get("id")))} <span class="mono">@{esc(hf["revision"])}</span></a>'


def code_link(code: dict) -> str:
    return f'<a href="https://github.com/{esc(code["repo"])}/commit/{esc(code["commit"])}">{esc(code["repo"])} <span class="mono">@{esc(code["commit"])}</span></a>'


def tag_list(tags: list[str], palette: dict) -> str:
    return "".join(
        f'<span class="tag" style="--tag-fill:{palette[tag][1]};--tag-ink:{palette[tag][2]}">{esc(tag)}</span>'
        for tag in tags
    )


def graph_svg(experiments: list[dict], bases: list[dict], palette: dict, format_parameters) -> str:
    nodes = {}
    edges = []
    for base in bases:
        nodes[base["id"]] = {"kind": "model", "model": base, "tags": []}
    for exp in experiments:
        nodes[exp["id"]] = {"kind": "experiment", "experiment": exp, "tags": exp.get("tags", [])}
        for inp in exp["inputs"]:
            edges.append((inp["model"], exp["id"], exp["status"], format_parameters(inp.get("parameters", {}))))
        for out in exp.get("outputs") or []:
            nodes[out["id"]] = {"kind": "model", "model": out, "tags": exp.get("tags", [])}
            edges.append((exp["id"], out["id"], exp["status"], ""))

    incoming = defaultdict(list)
    children = defaultdict(list)
    for source, target, _, _ in edges:
        incoming[target].append(source)
        children[source].append(target)
    remaining = {nid: len(incoming[nid]) for nid in nodes}
    ready = deque(sorted(nid for nid, count in remaining.items() if not count))
    levels = defaultdict(int)
    while ready:
        source = ready.popleft()
        for target in children[source]:
            levels[target] = max(levels[target], levels[source] + 1)
            remaining[target] -= 1
            if remaining[target] == 0:
                ready.append(target)
    layers = defaultdict(list)
    for nid in nodes:
        layers[levels[nid]].append(nid)
    height = max((len(layer) for layer in layers.values()), default=1) * 190 + 100
    width = (max(layers, default=0) + 1) * 380 + 60
    positions = {}
    for level, layer in sorted(layers.items()):
        layer.sort(key=lambda nid: (
            sum(positions[p][1] for p in incoming[nid]) / len(incoming[nid]) if incoming[nid] else 0,
            nid,
        ))
        for row, nid in enumerate(layer):
            positions[nid] = (40 + level * 380, (height - len(layer) * 190) / 2 + row * 190)

    svg = [f'<svg id="lineage-graph" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" data-width="{width}" data-height="{height}" role="img" aria-labelledby="graph-title graph-description">',
           '<title id="graph-title">Experiment lineage</title>',
           '<desc id="graph-description">Cylinders show checkpoints; rounded rectangles show experiments. Green arrows are accepted, red rejected, and dashed arrows in progress. Select an experiment to inspect its record.</desc>',
           '<defs>']
    for status, colour in COLOURS.items():
        svg.append(f'<marker id="arrow-{status}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{colour}"/></marker>')
    svg += ['</defs>', '<g id="graph-content" font-family="system-ui, sans-serif">']
    for source, target, status, label in edges:
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        x1 += 280
        y1 += 72
        y2 += 72
        middle = (x1 + x2) / 2
        dash = ' stroke-dasharray="9 7"' if status == "in_progress" else ""
        svg.append(f'<path class="graph-edge" data-status="{status}" d="M{x1},{y1} C{middle},{y1} {middle},{y2} {x2},{y2}" stroke="{COLOURS[status]}" stroke-width="3" fill="none"{dash} marker-end="url(#arrow-{status})"/>')
        if label:
            short = textwrap.shorten(label, width=34, placeholder="…")
            svg.append(f'<text x="{middle}" y="{(y1+y2)/2-12}" text-anchor="middle" fill="#475569" font-size="14" paint-order="stroke" stroke="#f8fafc" stroke-width="6" stroke-linejoin="round"><title>{esc(label)}</title>{esc(short)}</text>')
    for nid, node in nodes.items():
        x, y = positions[nid]
        tags = node["tags"]
        fill, stroke = palette[tags[0]][1:] if tags else ("#ffffff", "#cbd5e1")
        exp = node.get("experiment")
        activation = f' data-record="{esc(nid)}" tabindex="0" role="button" aria-label="Inspect {esc(nid)}"' if exp else ""
        svg.append(f'<g class="graph-node" data-kind="{node["kind"]}" transform="translate({x},{y})"{activation}><title>{esc(exp["title"] if exp else nid)}</title>')
        if exp:
            svg.append(f'<rect width="280" height="144" rx="16" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        else:
            svg.append(f'<path d="M0 18 A140 18 0 0 1 280 18 V126 A140 18 0 0 1 0 126 Z" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
            svg.append(f'<ellipse cx="140" cy="18" rx="140" ry="18" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        svg.append(f'<text x="20" y="{30 if exp else 57}" fill="#172e3d" font-size="18" font-weight="700">{esc(textwrap.shorten(nid, 25, placeholder="…"))}</text>')
        if exp:
            for index, line in enumerate(textwrap.wrap(exp["title"], width=29)[:2]):
                svg.append(f'<text x="20" y="{56+index*19}" fill="#334d5c" font-size="15">{esc(line)}</text>')
            svg.append(f'<circle cx="24" cy="105" r="4" fill="{COLOURS[exp["status"]]}"/><text x="35" y="110" fill="#475569" font-size="13">{exp["status"].replace("_", " ")}</text>')
            svg.append(f'<text x="20" y="130" fill="#475569" font-size="11">{esc(textwrap.shorten(", ".join(tags), 38, placeholder="…"))}</text>')
        else:
            model = node["model"]
            svg.append(f'<text x="20" y="78" fill="#64748b" font-size="13">checkpoint</text><text x="20" y="102" fill="#334d5c" font-size="15">@{esc(textwrap.shorten(str(model["hf"]["revision"]), 24, placeholder="…"))}</text>')
            svg.append(f'<text x="20" y="123" fill="#64748b" font-size="11">{esc(textwrap.shorten(model["hf"]["repo"], 36, placeholder="…"))}</text>')
        svg.append('</g>')
    svg += ['</g>', '</svg>']
    return "\n".join(svg)


def experiment_card(exp: dict, palette: dict, format_parameters) -> str:
    search = " ".join(str(exp.get(key, "")) for key in ("id", "title", "description", "tags", "method", "reject_reason"))
    inputs = "".join(f'<li>{hf_link(inp)}<small>{esc(format_parameters(inp.get("parameters", {})))}</small></li>' for inp in exp["inputs"])
    outputs = "".join(
        f'<li>{hf_link(out)}<small>{esc(", ".join(f"{key}: {value:g}" for key, value in out["scores"].items()))}</small>'
        f'<small>Evaluation: {code_link(out["eval"])}</small></li>'
        for out in exp.get("outputs") or []
    ) or '<li class="muted">No evaluated checkpoint yet</li>'
    config = exp.get("config", {})
    config_html = "".join(f'<dt>{esc(key)}</dt><dd><code>{esc(value)}</code></dd>' for key, value in config.items())
    reason = f'<div class="reject-reason"><strong>Rejection reason</strong><p>{esc(exp["reject_reason"])}</p></div>' if exp.get("reject_reason") else ""
    notes = f'<p class="record-notes">{esc(exp["notes"])}</p>' if exp.get("notes") else ""
    return f'''<details class="record" id="record-{esc(exp['id'])}" data-status="{exp['status']}" data-search="{esc(search.lower())}">
<summary><div class="record-top"><span class="record-id">{esc(exp['id'])}</span>{status_badge(exp['status'])}</div>
<h3>{esc(exp['title'])}</h3><p class="record-description">{esc(exp.get('description', ''))}</p><div class="tags">{tag_list(exp.get('tags', []), palette)}</div><span class="inspect">Inspect record <span aria-hidden="true">↗</span></span></summary>
<div class="record-body">{reason}<dl><dt>Method</dt><dd>{esc(exp['method'])}</dd><dt>Owner</dt><dd>{esc(exp['owner']['name'])} · {esc(exp['owner']['github'])}</dd><dt>Code</dt><dd>{code_link(exp['code'])}</dd>{config_html}</dl>
<h4>Inputs</h4><ul class="checkpoints">{inputs}</ul><h4>Outputs</h4><ul class="checkpoints">{outputs}</ul>{notes}<a class="source-link" href="{esc(exp['_source'])}">View source YAML ↗</a></div></details>'''


def render(config: dict, experiments: list[dict], rejected: list[dict], bases: list[dict], all_bases: list[dict], palette: dict, format_parameters, format_metadata) -> str:
    all_experiments = sorted(experiments + rejected, key=lambda exp: exp["id"])
    checkpoints = [(base, "external") for base in bases]
    checkpoints += [(out, exp["id"]) for exp in sorted(experiments, key=lambda exp: exp["id"]) for out in exp.get("outputs") or []]
    benchmarks = sorted({key for model, _ in checkpoints for key in model.get("scores", {})})
    rows = []
    for model, producer in checkpoints:
        scores = "".join(f'<td class="numeric">{model["scores"][key]:g}</td>' if key in model.get("scores", {}) else '<td class="numeric muted">—</td>' for key in benchmarks)
        evaluation = code_link(model["eval"]) if model.get("eval") else "—"
        rows.append(f'<tr><th scope="row">{hf_link(model)}</th><td>{esc(producer)}</td>{scores}<td>{evaluation}</td></tr>')
    status_counts = {status: sum(exp["status"] == status for exp in all_experiments) for status in COLOURS}
    replacements = {
        "PROJECT": esc(config["project_name"]),
        "BRANCH": esc(config["main_branch"]),
        "ACCEPTED": str(status_counts["accepted"]),
        "IN_PROGRESS": str(status_counts["in_progress"]),
        "REJECTED": str(status_counts["rejected"]),
        "CHECKPOINTS": str(len(checkpoints)),
        "GRAPH": graph_svg(all_experiments, all_bases, palette, format_parameters),
        "TAGS": tag_list(sorted(palette), palette),
        "RECORDS": "".join(experiment_card(exp, palette, format_metadata) for exp in all_experiments),
        "BENCHMARKS": "".join(f'<th scope="col" class="numeric">{esc(key)}</th>' for key in benchmarks),
        "ROWS": "".join(rows),
    }
    template = (Path(__file__).parent / "templates" / "lineage.html").read_text()
    # Substitute in one pass so metadata containing placeholder-like text stays literal.
    return re.sub(r"@@([A-Z_]+)@@", lambda match: replacements[match[1]], template)
