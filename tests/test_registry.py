import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import registry  # noqa: E402

PIPELINE, EXPERIMENTS = registry.load()


def errors_after(mutate):
    pipeline, experiments = copy.deepcopy(PIPELINE), copy.deepcopy(EXPERIMENTS)
    mutate(pipeline, {e["id"]: e for e in experiments}, experiments)
    return registry.validate(pipeline, experiments)


def test_demo_registry_is_valid():
    assert registry.validate(PIPELINE, EXPERIMENTS) == []


def test_unknown_input_model():
    errors = errors_after(lambda p, by_id, _: by_id["exp-002"]["inputs"][0].update(model="nope"))
    assert any("'nope' is not registered" in e for e in errors)


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
        out["scores"].pop("math")
        out.pop("eval")

    errors = errors_after(mutate)
    assert any("missing scores for math" in e for e in errors)
    assert any("needs eval.commit" in e for e in errors)


def test_soup_weights_must_sum_to_one():
    errors = errors_after(lambda p, by_id, _: by_id["exp-006"]["inputs"][0].update(weight=0.5))
    assert any("weights sum to 1.3" in e for e in errors)


def test_rejected_status_is_not_a_registry_status():
    errors = errors_after(lambda p, by_id, _: by_id["exp-010"].update(status="discarded"))
    assert any("unknown status 'discarded'" in e for e in errors)


def test_cycle_is_rejected():
    def mutate(p, by_id, _):
        by_id["exp-001"]["inputs"][0] = {
            "model": "rlhf2-v1",
            "hf": {"repo": "example-org/rlhf2-v1", "revision": "1e6c8b4"},
        }

    assert errors_after(mutate) == ["lineage contains a cycle"]


def test_main_must_exist():
    errors = errors_after(lambda p, by_id, _: p.update(main="ghost"))
    assert any("main 'ghost'" in e for e in errors)


def test_main_lineage_includes_every_soup_input():
    lineage = registry.main_lineage(PIPELINE, EXPERIMENTS)
    assert {"base", "soup-v1", "rlvr-a-v1", "rlvr-a-v2", "rlvr-b-v1", "exp-006"} <= lineage
    assert "exp-010" not in lineage
