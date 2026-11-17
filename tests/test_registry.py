import copy
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import registry  # noqa: E402

CONFIG, EXPERIMENTS = registry.load()


def errors_after(mutate):
    settings, experiments = copy.deepcopy(CONFIG), copy.deepcopy(EXPERIMENTS)
    mutate(settings, {e["id"]: e for e in experiments}, experiments)
    return registry.validate(settings, experiments)


def test_demo_registry_is_valid():
    assert registry.validate(CONFIG, EXPERIMENTS, registry.load_rejected()) == []


@pytest.mark.parametrize("settings", [
    None, [], {}, {"project_name": "Demo"}, {"main_branch": "main"},
    {"project_name": "", "main_branch": "main"},
    {"project_name": "Demo", "main_branch": " "},
    {"project_name": 1, "main_branch": "main"},
    {"project_name": "Demo", "main_branch": 1},
    {"project_name": "Demo", "main_branch": "bad branch"},
    {"project_name": "Demo", "main_branch": "main\nother"},
    {"project_name": "Demo", "main_branch": "main\0other"},
    {"project_name": "Demo", "main_branch": "main", "extra": True},
])
def test_invalid_project_configuration_is_reported(settings):
    assert any("config.yaml" in error for error in registry.validate(settings, EXPERIMENTS))


def test_project_uses_only_name_and_git_branch():
    assert set(CONFIG) == {"project_name", "main_branch"}
    settings = {"project_name": "Research runs", "main_branch": "release/latest"}
    assert registry.validate(settings, EXPERIMENTS) == []
    doc = registry.render(settings, EXPERIMENTS)
    assert doc.startswith("# Research runs — lineage")
    assert "**Main branch:** `release/latest`" in doc


def test_tags_need_no_configuration():
    settings, experiments = copy.deepcopy(CONFIG), copy.deepcopy(EXPERIMENTS)
    experiments[0]["tags"] = ["new arbitrary tag", "ablation"]
    experiments[1].pop("tags")
    experiments[2]["tags"] = []
    assert registry.validate(settings, experiments) == []
    doc = registry.render(settings, experiments)
    assert "new arbitrary tag, ablation" in doc
    assert "| Experiment | Tags | Method |" in doc
    # Table order follows IDs, regardless of tags or the input list's order.
    table = doc.split("## Experiments")[1].split("## Checkpoints")[0]
    assert table.index("| exp-001 |") < table.index("| exp-002 |")
    reversed_table = registry.render(settings, list(reversed(experiments))).split("## Experiments")[1].split("## Checkpoints")[0]
    assert table == reversed_table


@pytest.mark.parametrize("tags", [None, "rlvr", {}, [""], ["  "], [12], [True], [["nested"]]])
def test_tags_must_be_list_of_nonempty_strings(tags):
    errors = errors_after(lambda p, by_id, _: by_id["exp-005"].update(tags=tags))
    assert any("tags must be a list of nonempty strings" in e for e in errors)


def test_each_tag_gets_distinct_automatic_colour():
    experiments = [{"tags": [f"tag-{i}"]} for i in range(40)]
    styles = registry._tag_styles(experiments)
    assert len(styles) == 40
    assert len({fill for _, fill, _ in styles.values()}) == 40
    assert len({name for name, _, _ in styles.values()}) == 40
    assert styles == registry._tag_styles(list(reversed(experiments)))


def test_first_tag_colours_experiment_and_outputs_with_all_tags_in_legend():
    settings, experiments = copy.deepcopy(CONFIG), copy.deepcopy(EXPERIMENTS)
    exp = next(e for e in experiments if e["id"] == "exp-006")
    exp["tags"] = ["custom primary", 'other | "tag"']
    graph = registry.render_mermaid(settings, experiments)
    name, fill, stroke = registry._tag_styles(experiments)["custom primary"]
    assert f"classDef {name} fill:{fill},stroke:{stroke}" in graph
    assignment = next(line for line in graph.splitlines() if line.startswith("  class ") and line.endswith(f" {name}"))
    assert "exp_006" in assignment and "soup_v1" in assignment
    assert '["custom primary"]' in graph
    assert '["other #124; #34;tag#34;"]' in graph
    assert "custom primary, other #124; #34;tag#34;" in graph


def test_untagged_registry_renders_without_legend():
    settings, experiments = copy.deepcopy(CONFIG), copy.deepcopy(EXPERIMENTS)
    for exp in experiments:
        exp.pop("tags")
    assert registry.validate(settings, experiments) == []
    graph = registry.render_mermaid(settings, experiments)
    assert "tag_legend" not in graph
    assert "classDef default fill:#ffffff" in graph
    assert '| exp-006 | – | weighted_average |' in registry.render(settings, experiments)


def test_tag_does_not_control_method_validation():
    errors = errors_after(lambda p, by_id, _: by_id["exp-006"].update(tags=["arbitrary"], method=""))
    assert any("nonempty string 'method'" in e for e in errors)


def test_loading_preserves_arbitrary_input_parameters(tmp_path):
    (tmp_path / "config.yaml").write_text("{}")
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "example.yaml").write_text(
        "code: {commit: 1234567}\n"
        "inputs:\n"
        "  - parameters: {revision: 12, nested: {commit: 34}}\n"
    )
    _, experiments = registry.load(tmp_path)
    assert experiments[0]["code"]["commit"] == "1234567"
    assert experiments[0]["inputs"][0]["parameters"] == {"revision": 12, "nested": {"commit": 34}}


def test_new_input_without_producer_is_external():
    errors = errors_after(lambda p, by_id, _: by_id["exp-002"]["inputs"][0].update(model="nope"))
    assert errors == []


def test_external_input_is_discovered_without_evaluation_metadata():
    settings, experiments = copy.deepcopy(CONFIG), copy.deepcopy(EXPERIMENTS)
    experiments[0]["inputs"] = [{"model": "external-base", "hf": {"repo": "org/base", "revision": "abc1234"}}]
    assert registry.validate(settings, experiments) == []
    assert registry.external_models(experiments) == [
        {"id": "external-base", "hf": {"repo": "org/base", "revision": "abc1234"}}
    ]
    graph = registry.render_mermaid(settings, experiments)
    assert 'external_base(["external-base<br/>@abc1234"])' in graph
    doc = registry.render(settings, experiments)
    assert "| `external-base` | external |" in doc
    assert "| – | – | – | – | – | exp-001 |" in doc


def test_repeated_external_inputs_must_have_same_checkpoint():
    def mutate(p, by_id, _):
        by_id["exp-002"]["inputs"] = [{"model": "base", "hf": {"repo": "other/base", "revision": "bad"}}]
    assert any("but it is registered as" in e for e in errors_after(mutate))


def test_external_input_still_needs_pinned_checkpoint():
    errors = errors_after(lambda p, by_id, _: by_id["exp-001"]["inputs"][0].pop("hf"))
    assert any("needs hf.repo and hf.revision" in e for e in errors)


def test_benchmarks_are_discovered_and_missing_values_are_dashes():
    settings, experiments = copy.deepcopy(CONFIG), copy.deepcopy(EXPERIMENTS)
    experiments[0]["outputs"][0]["scores"] = {"new-benchmark": 0.7}
    assert registry.validate(settings, experiments) == []
    doc = registry.render(settings, experiments)
    assert "| chat | code | math | new-benchmark | safety |" in doc
    row = next(line for line in doc.splitlines() if line.startswith("| `sft-v1` |"))
    assert "| – | – | – | 0.7 | – |" in row


@pytest.mark.parametrize("value", [True, "0.5", None])
def test_discovered_scores_must_be_numeric(value):
    errors = errors_after(lambda p, by_id, _: by_id["exp-001"]["outputs"][0]["scores"].update(new=value))
    assert any("score 'new' is not a number" in e for e in errors)


def test_empty_registry_has_valid_checkpoint_table():
    assert registry.validate(CONFIG, []) == []
    doc = registry.render(CONFIG, [])
    assert "| Model | Produced by | Checkpoint | Eval code | Used by |" in doc


def test_parent_revision_must_match_registered_checkpoint():
    errors = errors_after(lambda p, by_id, _: by_id["exp-007"]["inputs"][0]["hf"].update(revision="0000000"))
    assert any("points to example-org/soup-v1@0000000" in e for e in errors)


def test_parent_needs_hf_pointer():
    errors = errors_after(lambda p, by_id, _: by_id["exp-002"]["inputs"][0].pop("hf"))
    assert any("input 'sft-v1' needs hf.repo and hf.revision" in e for e in errors)


def test_duplicate_model_id():
    errors = errors_after(lambda p, by_id, _: by_id["exp-005"]["outputs"][0].update(id="sft-v1"))
    assert any("already defined" in e for e in errors)


def test_missing_revision():
    errors = errors_after(lambda p, by_id, _: by_id["exp-001"]["outputs"][0]["hf"].pop("revision"))
    assert any("hf.revision" in e for e in errors)


def test_owner_needs_email():
    errors = errors_after(lambda p, by_id, _: by_id["exp-001"]["owner"].pop("email"))
    assert any("owner needs name, email and github" in e for e in errors)


def test_scores_and_eval_commit_required():
    def mutate(p, by_id, _):
        out = by_id["exp-003"]["outputs"][0]
        out.pop("scores")
        out.pop("eval")

    errors = errors_after(mutate)
    assert any("needs nonempty scores" in e for e in errors)
    assert any("needs eval.commit" in e for e in errors)
    assert any("needs eval.repo" in e for e in errors)


def test_evaluation_repo_is_validated_and_rendered_per_checkpoint():
    errors = errors_after(lambda p, by_id, _: by_id["exp-003"]["outputs"][0]["eval"].update(repo="invalid"))
    assert any("eval.repo must be a GitHub repository" in e for e in errors)
    settings, experiments = copy.deepcopy(CONFIG), copy.deepcopy(EXPERIMENTS)
    evaluation = experiments[0]["outputs"][0]["eval"]
    evaluation.update(repo="other-org/evaluation", commit="abc1234")
    assert "https://github.com/other-org/evaluation/commit/abc1234" in registry.render(settings, experiments)


def test_aggregation_metadata_has_no_method_specific_rules():
    def mutate(p, by_id, _):
        exp = by_id["exp-006"]
        exp["inputs"][0]["parameters"]["weight"] = 0.5

    assert errors_after(mutate) == []

    def arbitrary_method(p, by_id, _):
        exp = by_id["exp-006"]
        exp["method"] = "custom_merge"
        exp["inputs"] = exp["inputs"][:1]
        exp["inputs"][0]["parameters"] = {"role": "base", "layers": {"range": [0, 12]}}

    assert errors_after(arbitrary_method) == []


@pytest.mark.parametrize("method", [None, "", "  ", 12, {}, []])
def test_experiment_requires_nonempty_method(method):
    errors = errors_after(lambda p, by_id, _: by_id["exp-006"].update(method=method))
    assert any("nonempty string 'method'" in e for e in errors)


def test_experiment_requires_method_field():
    errors = errors_after(lambda p, by_id, _: by_id["exp-006"].pop("method"))
    assert any("nonempty string 'method'" in e for e in errors)


@pytest.mark.parametrize("parameters", [None, [], "weight=0.2", 0.2, True])
def test_input_parameters_must_be_dictionary(parameters):
    # Input metadata is supported for all experiments, including ordinary training.
    errors = errors_after(lambda p, by_id, _: by_id["exp-005"]["inputs"][0].update(parameters=parameters))
    assert any("parameters must be a dictionary" in e for e in errors)


@pytest.mark.parametrize("config", [
    {"file": "configs/rlvr.yaml"},
    {"cli": "--seed 42"},
    {"file": "configs/rlvr.yaml", "cli": "--seed 42"},
])
def test_config_accepts_file_cli_or_both(config):
    assert errors_after(lambda p, by_id, _: by_id["exp-005"].update(config=config)) == []
    settings, experiments = copy.deepcopy(CONFIG), copy.deepcopy(EXPERIMENTS)
    exp = next(e for e in experiments if e["id"] == "exp-005")
    exp["config"] = config
    doc = registry.render(settings, experiments)
    for key, value in config.items():
        assert f"{key}: {value}" in doc


@pytest.mark.parametrize("config", [
    None, [], "configs/rlvr.yaml", {}, {"other": "x"},
    {"file": ""}, {"cli": "  "}, {"file": 42},
    {"cli": ["--seed", "42"]}, {"file": "config.yaml", "cli": None},
    {"file": "config.yaml", "unknown": "value"},
])
def test_malformed_config_is_rejected(config):
    errors = errors_after(lambda p, by_id, _: by_id["exp-005"].update(config=config))
    assert any("config" in e for e in errors)


def test_config_is_optional():
    assert errors_after(lambda p, by_id, _: by_id["exp-005"].pop("config")) == []


def test_parameters_and_method_render_in_graph_and_table():
    settings, experiments = copy.deepcopy(CONFIG), copy.deepcopy(EXPERIMENTS)
    exp = next(e for e in experiments if e["id"] == "exp-006")
    exp["inputs"][0]["parameters"] = {"weight": 0.2, "layers": {"range": [0, 12]}}
    exp["inputs"][1]["parameters"] = {}
    exp["inputs"][2].pop("parameters")
    assert registry.validate(settings, experiments) == []
    graph = registry.render_mermaid(settings, experiments)
    assert 'rlvr_a_v1 -->|"weight: 0.2, layers: #123;#34;range#34;: #91;0, 12#93;#125;"| exp_006' in graph
    assert "rlvr_a_v2 --> exp_006" in graph
    assert "rlvr_b_v1 --> exp_006" in graph
    assert "<br/>weighted_average<br/>" in graph
    doc = registry.render(settings, experiments)
    assert 'weight: 0.2, layers: {"range": [0, 12]}' in doc
    assert "| weight_aggregation, souping | weighted_average |" in doc


def test_metadata_rendering_escapes_graph_and_table_syntax():
    settings, experiments = copy.deepcopy(CONFIG), copy.deepcopy(EXPERIMENTS)
    exp = next(e for e in experiments if e["id"] == "exp-006")
    text = 'a|b "quoted" <tag> & `code` #34;\nnext'
    exp["method"] = text
    exp["inputs"][0]["parameters"] = {"label": text}
    exp["config"] = {"cli": text}
    graph = registry.render_mermaid(settings, experiments)
    assert 'label: a#124;b #34;quoted#34; #60;tag#62; #38; #96;code#96; #35;34;#10;next' in graph
    doc = registry.render(settings, experiments)
    escaped = 'a&#124;b "quoted" &lt;tag&gt; &amp; &#96;code&#96; #34;<br/>next'
    assert f"label: {escaped}" in doc
    assert f"cli: {escaped}" in doc
    assert f"| {escaped} |" in doc


@pytest.mark.parametrize("status", ["planned", "running", "paused", "inconclusive", "discarded"])
def test_unknown_status_is_not_a_registry_status(status):
    errors = errors_after(lambda p, by_id, _: by_id["exp-010"].update(status=status))
    assert any(f"unknown status '{status}'" in e for e in errors)


def test_rejected_status_requires_rejected_directory():
    errors = errors_after(lambda p, by_id, _: by_id["exp-010"].update(status="rejected"))
    assert any("move rejected experiments to rejected/" in e for e in errors)


def test_rejection_workflow_moves_yaml_and_preserves_record(tmp_path):
    shutil.copy(registry.ROOT / "config.yaml", tmp_path / "config.yaml")
    shutil.copytree(registry.ROOT / "experiments", tmp_path / "experiments")
    (tmp_path / "rejected").mkdir()
    original = tmp_path / "experiments" / "exp-010-rlvr-d.yaml"
    destination = tmp_path / "rejected" / original.name
    original.rename(destination)
    destination.write_text(destination.read_text().replace("status: in_progress", "status: rejected") + "\nreject_reason: Poor results.\n")
    # Templates are not records, and are ignored in both folders.
    (tmp_path / "rejected" / "_template.yaml").write_text("status: rejected")
    settings, experiments = registry.load(tmp_path)
    rejected = registry.load_rejected(tmp_path)
    assert len(rejected) == 1
    assert rejected[0]["id"] == "exp-010"
    assert registry.validate(settings, experiments, rejected) == []
    graph = registry.render_mermaid(settings, experiments)
    assert "exp_010" not in graph
    doc = registry.render(settings, experiments, rejected)
    active, archive = doc.split("## Rejected experiments")
    active_tables = active.split("## Experiments")[1]
    assert "exp-010" not in active_tables
    assert "[exp-010](rejected/exp-010-rlvr-d.yaml)" in archive
    assert "Poor results." in archive
    assert "issues" not in doc


def rejected_copy(eid):
    settings, experiments = copy.deepcopy(CONFIG), copy.deepcopy(EXPERIMENTS)
    exp = next(e for e in experiments if e["id"] == eid)
    experiments.remove(exp)
    exp["status"] = "rejected"
    exp["_source"] = f"rejected/{eid}.yaml"
    return settings, experiments, [exp]


def test_rejected_directory_requires_rejected_status():
    settings, experiments, rejected = rejected_copy("exp-010")
    rejected[0]["status"] = "accepted"
    assert any("files in rejected/ need status: rejected" in e for e in registry.validate(settings, experiments, rejected))


def test_rejected_records_are_validated():
    settings, experiments, rejected = rejected_copy("exp-010")
    rejected[0]["inputs"][0]["hf"]["revision"] = "bad"
    rejected[0]["method"] = ""
    errors = registry.validate(settings, experiments, rejected)
    assert any("but it is registered as" in e for e in errors)
    assert any("nonempty string 'method'" in e for e in errors)


def test_active_experiments_cannot_use_rejected_checkpoints():
    settings, experiments, rejected = rejected_copy("exp-005")
    errors = registry.validate(settings, experiments, rejected)
    assert any("exp-006: input 'rlvr-b-v1' belongs to a rejected experiment" in e for e in errors)


def test_rejected_outputs_appear_in_overview_but_stay_outside_active_tables():
    settings, experiments, rejected = rejected_copy("exp-010")
    rejected[0]["reject_reason"] = "Worse | scores\nNeeds review"
    rejected[0]["notes"] = "Separate context, not the rejection reason."
    assert registry.validate(settings, experiments, rejected) == []
    doc = registry.render(settings, experiments, rejected)
    active, archive = doc.split("## Rejected experiments")
    active_tables = active.split("## Experiments")[1]
    assert "exp-010" not in active_tables and "rlvr-d-v1" not in active_tables
    assert "exp_010 --> rlvr_d_v1" in active
    assert "[exp-010](rejected/exp-010.yaml)" in archive
    assert "Worse &#124; scores<br/>Needs review" in archive
    assert "Separate context" not in archive


def test_arrow_styles_follow_each_experiments_status():
    rejected = registry.load_rejected()
    graph = registry.render_mermaid(CONFIG, EXPERIMENTS, rejected)
    edge_lines = [line.strip() for line in graph.splitlines() if " -->" in line]
    styles = {}
    for indices, style in re.findall(r"^  linkStyle ([\d,]+) (.+)$", graph, flags=re.MULTILINE):
        for index in map(int, indices.split(",")):
            assert index not in styles
            styles[index] = style
    assert set(styles) == set(range(len(edge_lines)))
    expectations = {}
    for exp in EXPERIMENTS + rejected:
        for inp in exp["inputs"]:
            expectations[(inp["model"].replace("-", "_"), exp["id"].replace("-", "_"))] = exp["status"]
        for out in exp.get("outputs", []):
            expectations[(exp["id"].replace("-", "_"), out["id"].replace("-", "_"))] = exp["status"]
    colours = {"accepted": "#16a34a", "rejected": "#dc2626", "in_progress": "#64748b"}
    for index, line in enumerate(edge_lines):
        status = expectations[(line.split()[0], line.rsplit(" ", 1)[1])]
        assert f"stroke:{colours[status]}" in styles[index]
        assert ("stroke-dasharray:" in styles[index]) == (status == "in_progress")


def test_rejected_records_can_reference_other_rejected_checkpoints():
    settings, experiments, rejected = rejected_copy("exp-007")
    for eid in ("exp-008", "exp-013"):
        dependent = next(e for e in experiments if e["id"] == eid)
        experiments.remove(dependent)
        dependent["status"] = "rejected"
        rejected.append(dependent)
    assert registry.validate(settings, experiments, rejected) == []


def test_rejected_and_active_ids_must_be_unique():
    settings, experiments, rejected = rejected_copy("exp-010")
    rejected[0]["id"] = "exp-001"
    assert any("duplicate experiment id" in e for e in registry.validate(settings, experiments, rejected))


def test_rejected_directory_is_optional(tmp_path):
    assert registry.load_rejected(tmp_path) == []


def test_code_repo_must_be_github_owner_and_name():
    for bad in ("https://github.com/example-org/training", "training"):
        errors = errors_after(lambda p, by_id, _: by_id["exp-004"]["code"].update(repo=bad))
        assert any("must be a GitHub repository as owner/name" in e for e in errors)


def test_cycle_is_rejected():
    def mutate(p, by_id, _):
        by_id["exp-001"]["inputs"][0] = {
            "model": "rlhf2-v1",
            "hf": {"repo": "example-org/rlhf2-v1", "revision": "1e6c8b4"},
        }

    assert errors_after(mutate) == ["lineage contains a cycle"]


def test_graph_includes_every_aggregation_input():
    graph = registry.render_mermaid(CONFIG, EXPERIMENTS)
    for mid in ("base", "soup_v1", "rlvr_a_v1", "rlvr_a_v2", "rlvr_b_v1", "exp_006"):
        assert mid in graph


def test_html_contains_all_records_and_offline_svg_with_status_edges():
    rejected = registry.load_rejected()
    doc = registry.render_html(CONFIG, EXPERIMENTS, rejected)
    svg_source = re.search(r"<svg .*?</svg>", doc, flags=re.DOTALL)[0]
    svg = ET.fromstring(svg_source)
    ns = {"s": "http://www.w3.org/2000/svg"}
    nodes = svg.findall('.//s:g[@data-record]', ns)
    assert {node.attrib["data-record"] for node in nodes} == {exp["id"] for exp in EXPERIMENTS + rejected}
    edges = svg.findall('.//s:path[@class="graph-edge"]', ns)
    for status, colour in {"accepted": "#168654", "rejected": "#d34a4a", "in_progress": "#64748b"}.items():
        styled = [edge for edge in edges if edge.attrib["data-status"] == status]
        assert styled and all(edge.attrib["stroke"] == colour for edge in styled)
        assert all(("stroke-dasharray" in edge.attrib) == (status == "in_progress") for edge in styled)
    assert "weight: 0.3333333333333333" not in svg_source
    assert "weight: 0.333" in svg_source
    assert "weight: 0.3333333333333333" in doc  # Record details retain full precision.
    for exp in EXPERIMENTS + rejected:
        assert f'id="record-{exp["id"]}"' in doc
        if exp.get("reject_reason"):
            assert exp["reject_reason"] in doc
    tbody = doc.split("<tbody>")[1].split("</tbody>")[0]
    assert "soup-uniform-v1" not in tbody and "rlvr-c-v1" not in tbody
    assert "class=\"numeric muted\">—" in tbody
    assert '<script src=' not in doc and 'import mermaid' not in doc


def test_html_escapes_metadata_and_does_not_interpret_template_tokens():
    experiments = copy.deepcopy(EXPERIMENTS)
    payload = '</script><script>alert("x")</script>'
    experiments[0]["title"] = payload
    experiments[0]["description"] = payload
    doc = registry.render_html({**CONFIG, "project_name": "Project @@GRAPH@@"}, experiments)
    class ScriptCounter(HTMLParser):
        scripts = 0
        def handle_starttag(self, tag, attrs):
            if tag == "script":
                self.scripts += 1
    parser = ScriptCounter()
    parser.feed(doc)
    assert parser.scripts == 1
    assert payload not in doc
    assert '&lt;/script&gt;&lt;script&gt;alert(&quot;x&quot;)' in doc
    assert '<h1>Project @@GRAPH@@</h1>' in doc


def test_empty_registry_has_usable_html():
    doc = registry.render_html(CONFIG, [])
    assert '<tbody></tbody>' in doc
    assert 'id="empty-results"' in doc
    assert 'data-width="440"' in doc
    assert not re.search(r"@@(?:PROJECT|RECORDS|ROWS)@@", doc)


def test_workflow_reads_config_and_publishes_on_custom_branch(tmp_path):
    workflow = yaml.safe_load((registry.ROOT / ".github/workflows/registry.yml").read_text())
    steps = workflow["jobs"]["registry"]["steps"]
    config_step = next(step for step in steps if step.get("id") == "config")
    publish = next(step for step in steps if step.get("name") == "Publish lineage")
    assert "steps.config.outputs.main_branch" in publish["if"]
    assert "github.event_name == 'push'" in publish["if"]
    assert workflow["on"]["push"] is None  # No hard-coded branch filter.

    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copytree(registry.ROOT / "scripts", repo / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(registry.ROOT / "experiments", repo / "experiments")
    shutil.copytree(registry.ROOT / "rejected", repo / "rejected")
    branch = "release/latest"
    (repo / "config.yaml").write_text(yaml.safe_dump({"project_name": "Test project", "main_branch": branch}))
    (repo / "LINEAGE.md").write_text("Needs regeneration\n")
    remote = tmp_path / "origin.git"

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.PIPE).strip()

    git("init", "--initial-branch", branch)
    git("config", "user.name", "Registry test")
    git("config", "user.email", "registry-test@example.org")
    git("config", "commit.gpgsign", "false")
    git("config", "core.hooksPath", "/dev/null")
    git("add", ".")
    git("commit", "-m", "Initial records")
    git("init", "--bare", str(remote))
    git("remote", "add", "origin", str(remote))
    git("push", "origin", branch)

    output = tmp_path / "outputs.txt"
    env = {
        **os.environ,
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "GITHUB_OUTPUT": str(output),
        "MAIN_BRANCH": branch,
    }
    subprocess.run(["bash", "-e", "-c", config_step["run"]], cwd=repo, env=env, check=True, capture_output=True)
    assert output.read_text() == f"main_branch={branch}\n"
    subprocess.run(["bash", "-e", "-c", publish["run"]], cwd=repo, env=env, check=True, capture_output=True)
    assert "**Main branch:** `release/latest`" in (repo / "LINEAGE.md").read_text()
    head = git("rev-parse", "HEAD")
    assert git("rev-parse", f"refs/remotes/origin/{branch}") == head
    assert "<title>Test project · Model lineage</title>" in (repo / "LINEAGE.html").read_text()
    assert set(git("show", "--pretty=", "--name-only", "HEAD").splitlines()) == {"LINEAGE.md", "LINEAGE.html"}
    # A subsequent publication with identical metadata creates no extra commit.
    subprocess.run(["bash", "-e", "-c", publish["run"]], cwd=repo, env=env, check=True, capture_output=True)
    assert git("rev-parse", "HEAD") == head
