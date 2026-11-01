import json
import math

import pytest

from geotech_rl import grader, instance, quality, scenario, solver


class _NoInstances:
    def __init__(self, max_workers):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def map(self, function, jobs, chunksize):
        return [None for _ in jobs]


@pytest.mark.parametrize("level", scenario.LEVELS)
def test_filtered_instance_has_margin(level, tmp_path):
    built = instance.build(4, level, tmp_path, draws=12)
    assert built.betas["reference"] >= instance.BETA_MIN
    assert all(v >= instance.BETA_MIN for v in built.betas.values())


@pytest.mark.parametrize("level", scenario.LEVELS)
@pytest.mark.parametrize("split", ["heldout", "ood"])
def test_splits_build(level, split, tmp_path):
    built = instance.build(quality.SEED_BASE[split] + 1, level, tmp_path, split)
    assert grader.grade(built.truth, built.scenario.fs_required, solver.solve(tmp_path))["reward"] == 1.0


@pytest.mark.parametrize("required", [1.5, 2.0, 2.5])
def test_every_required_factor_is_reachable(required, tmp_path):
    """Regression: a regex that swallowed the final period rejected every factor but 2.0, silently."""
    (tmp_path / "README.md").write_text(f"All poles are 9 m long.\nRequired safety factor: {required:g}.\n", encoding="utf-8")
    assert solver.read_project(tmp_path)["fs_required"] == required
    seen = {instance.build(s, 1, tmp_path / str(s)).scenario.fs_required for s in range(30)}
    assert seen == {1.5, 2.0, 2.5}


def test_ood_sits_outside_training_ranges():
    train, ood = scenario.SPLITS["train"], scenario.SPLITS["ood"]
    for key in ("phi", "n", "density"):
        assert ood[key][0] >= train[key][1]
    assert not set(ood["lengths"]) & set(train["lengths"])
    assert not set(ood["required"]) & set(train["required"])


def test_a_level_that_rejects_everything_stops_instead_of_looping(monkeypatch):
    """OOD seeds that the mechanism could not bite left eight workers spinning for two hours."""
    monkeypatch.setattr(quality, "ProcessPoolExecutor", _NoInstances)
    with pytest.raises(RuntimeError, match="not biting"):
        quality.build_manifest("ood", 4, 2, 0, 1)


def test_seed_bases_do_not_overlap():
    assert quality.SEED_BASE["train"] + 10 ** 5 < quality.SEED_BASE["heldout"] < quality.SEED_BASE["ood"]


def test_known_attempt_rebuilds_same_instance(tmp_path):
    first = instance.build(9, 3, tmp_path / "a")
    second = instance.build(9, 3, tmp_path / "b", attempt=first.scenario.attempt)
    assert first.truth == second.truth


@pytest.mark.parametrize("answer", [
    None, [], "text", {"L-01": None}, {"L-01": {"fs": True, "refound": True}},
    {"L-01": {"fs": float("nan"), "refound": True}}, {"L-01": {"fs": float("inf"), "refound": True}},
    {"L-01": {"fs": 2.0}}, {"L-01": {"fs": 2.0, "FS": 2.0, "refound": True}},
    {"L-01": {"fs": "2.0", "refound": True}}, {"L-01": {"fs": 2.0, "refound": "yes"}},
])
def test_malformed_answers_score_zero(answer):
    truth = {"L-01": {"fs": 2.0, "refound": False}}
    assert grader.grade(truth, 1.5, answer)["reward"] == 0.0


def test_partial_answer_scores_zero_reward():
    truth = {"L-01": {"fs": 2.0, "refound": False}, "L-02": {"fs": 1.0, "refound": True}}
    result = grader.grade(truth, 1.5, {"L-01": {"fs": 2.0, "refound": False}})
    assert result["reward"] == 0.0 and result["partial"] == 0.5


def test_extra_poles_do_not_break_grading():
    truth = {"L-01": {"fs": 2.0, "refound": False}}
    answer = {"L-01": {"fs": 2.0, "refound": False}, "L-99": {"fs": 1.0, "refound": True}}
    assert grader.grade(truth, 1.5, answer)["reward"] == 1.0


def test_worst_error_is_infinite_for_missing_pole():
    assert math.isinf(grader.worst_error({"L-01": {"fs": 2.0, "refound": False}}, {}))


def test_nan_in_json_is_rejected():
    truth = {"L-01": {"fs": 2.0, "refound": False}}
    assert grader.grade(truth, 1.5, json.loads('{"L-01": {"fs": NaN, "refound": false}}'))["reward"] == 0.0


def test_duplicate_keys_count_as_no_answer():
    assert grader.loads('{"L-01": {"fs": 1}, "L-01": {"fs": 2}}') == {}
    assert grader.loads("not json") == {} and grader.loads("") == {}


@pytest.mark.parametrize("level", scenario.LEVELS)
def test_other_valid_methods_pass(level, tmp_path):
    """A right engineer need not reproduce the reference: other model and other reading of the load both pass."""
    built = instance.build(2, level, tmp_path)
    for route in solver.VALID_ROUTES:
        assert built.routes[route]["reward"] == 1.0


@pytest.mark.parametrize("level", [k for k, v in scenario.LEVELS.items() if not v.hint])
def test_upper_levels_do_not_name_the_governing_state(level, tmp_path):
    instance.build(3, level, tmp_path)
    text = (tmp_path / "README.md").read_text(encoding="utf-8").lower()
    assert not any(word in text for word in ("soaked", "rainy", "cracks", "do not measure"))


@pytest.mark.parametrize("level", [k for k, v in scenario.LEVELS.items() if not v.warmup])
def test_every_declared_mechanism_changes_the_answer(level, tmp_path):
    """Before this gate, ignoring the units or keeping a cracked test solved 100% of the upper levels."""
    declared = instance.mechanisms(scenario.LEVELS[level])
    assert declared
    built = [instance.build(seed, level, tmp_path / str(seed)) for seed in range(30)]
    plain = [b for b in built if not b.scenario.control]
    assert len(plain) / len(built) >= 1.0 - quality.CONTROL_SHARE_MAX
    assert not [route for route in declared for b in plain if b.routes[route]["reward"] > 0.0]


def test_the_null_table_reports_a_leak(tmp_path, monkeypatch):
    rows = [dict(seed=s, level=4, split="train", attempt=instance.build(s, 4, tmp_path / str(s)).scenario.attempt) for s in range(6)]
    assert quality.null_table(rows)["leaking"] == []
    monkeypatch.setattr(instance, "mechanisms", lambda level: ("ignore_units",))
    assert quality.null_table(rows)["leaking"] == ["ignore_units"]


def test_diagnosis_names_the_route_an_answer_followed(tmp_path):
    built = instance.build(1, 5, tmp_path)
    assert built.scenario.control == "" or built.scenario.control
    assert instance.diagnose(tmp_path, solver.solve(tmp_path)) == "reference"
    if "keep_cracked" in instance.must_fail(built.scenario):
        assert instance.diagnose(tmp_path, solver.solve(tmp_path, "keep_cracked")) == "keep_cracked"
    assert instance.diagnose(tmp_path, {}) == "other"


@pytest.mark.parametrize("level", [k for k, v in scenario.LEVELS.items() if v.warmup])
def test_warmup_levels_are_marked_as_having_no_decision(level, tmp_path):
    built = instance.build(1, level, tmp_path)
    wrong = [r for r in ("all_tests", "dry_only", "keep_cracked", "ignore_units") if built.routes[r]["reward"] == 1.0]
    assert len(wrong) == 4
