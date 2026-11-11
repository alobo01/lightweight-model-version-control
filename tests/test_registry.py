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


def test_duplicate_model_id():
    errors = errors_after(lambda p, by_id, _: by_id["exp-005"]["outputs"][0].update(id="sft-v1"))
    assert any("already defined" in e for e in errors)


def test_discarded_needs_reason():
    errors = errors_after(lambda p, by_id, _: by_id["exp-009"].pop("reason"))
    assert any("need a 'reason'" in e for e in errors)


def test_missing_revision():
    errors = errors_after(lambda p, by_id, _: by_id["exp-001"]["outputs"][0]["hf"].pop("revision"))
    assert any("hf.revision" in e for e in errors)


def test_cycle_is_rejected():
    errors = errors_after(lambda p, by_id, _: by_id["exp-001"]["inputs"][0].update(model="rlhf2-v1"))
    assert errors == ["lineage contains a cycle"]


def test_main_must_exist():
    errors = errors_after(lambda p, by_id, _: p.update(main="ghost"))
    assert any("main 'ghost'" in e for e in errors)


def test_main_lineage_skips_unused_branches():
    lineage = registry.main_lineage(PIPELINE, EXPERIMENTS)
    assert {"base", "soup-v1", "rlvr-a-v2", "rlvr-b-v1", "exp-006"} <= lineage
    assert not {"rlvr-a-v1", "rlvr-c-v1", "exp-009", "exp-010"} & lineage
