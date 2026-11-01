import collections
import filecmp
import json
import re

import pytest

from geotech_rl import grader, instance, solver
from geotech_rl.scenario import LEVELS

SEEDS = range(20)
LEAKS = re.compile(r"control instance|trap|suction|caved|bait|swap", re.I)


def files(root):
    return sorted(p for p in root.rglob("*") if p.is_file())


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_reference_passes_and_empty_answer_fails(level, seed, tmp_path):
    built = instance.build(seed, level, tmp_path)
    required = built.scenario.fs_required
    assert grader.grade(built.truth, required, solver.solve(tmp_path))["reward"] == 1.0
    assert grader.grade(built.truth, required, {})["reward"] == 0.0


@pytest.mark.parametrize("level", LEVELS)
def test_same_seed_gives_same_files(level, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    instance.build(3, level, a)
    instance.build(3, level, b)
    names = [p.relative_to(a) for p in files(a)]
    assert names == [p.relative_to(b) for p in files(b)]
    assert all(filecmp.cmp(a / n, b / n, shallow=False) for n in names)


@pytest.mark.parametrize("level", LEVELS)
def test_mechanism_position_varies(level, tmp_path):
    """The defect must not sit in the same test across seeds, or the model would memorise it."""
    wet = collections.Counter()
    for seed in SEEDS:
        built = instance.build(seed, level, tmp_path / str(seed))
        wet.update(t.id for t in built.scenario.tests if t.wet)
    assert len(wet) >= min(3, LEVELS[level].n_tests)


@pytest.mark.parametrize("level", [k for k, v in LEVELS.items() if not v.field])
def test_clean_levels_open_with_a_plain_read(level, tmp_path):
    """Comment lines stopped 151 of 175 failed cells on the first rung. Field levels keep them on purpose."""
    pandas = pytest.importorskip("pandas")
    instance.build(1, level, tmp_path)
    for path in tmp_path.rglob("*.csv"):
        assert len(pandas.read_csv(path).columns) >= 2, path.name


@pytest.mark.parametrize("level", LEVELS)
def test_records_do_not_name_the_mechanism(level, tmp_path):
    instance.build(1, level, tmp_path)
    for path in files(tmp_path):
        assert not LEAKS.search(path.read_text(encoding="utf-8")), path


@pytest.mark.parametrize("level", [k for k, v in LEVELS.items() if v.field])
def test_field_levels_keep_the_test_data_inside_each_sheet(level, tmp_path):
    instance.build(1, level, tmp_path)
    assert not (tmp_path / "pull_tests.csv").exists()
    sheet = sorted((tmp_path / "pull_tests").glob("*.csv"))[0].read_text(encoding="utf-8")
    assert sheet.startswith("# lateral pull test")


@pytest.mark.parametrize("level", [k for k, v in LEVELS.items() if v.crack])
def test_the_crack_sits_on_a_governing_test(level, tmp_path):
    """On a dry test the crack changed nothing: the right route had already set that test aside."""
    for seed in range(10):
        tests = instance.build(seed, level, tmp_path / str(seed)).scenario.tests
        assert all(t.wet for t in tests if t.remark)


@pytest.mark.parametrize("level", [k for k, v in LEVELS.items() if not v.all_wet])
def test_a_dry_test_stopped_early_keeps_the_lowest_load_from_being_the_answer(level, tmp_path):
    """Soaked tests are the weakest, so the lowest constant used to pass 90% of these levels with no look at the rain."""
    built = instance.build(1, level, tmp_path)
    short = [t for t in built.scenario.tests if t.short_ratio]
    assert len(short) == 1 and not short[0].wet
    sheet = (tmp_path / "pull_tests" / f"{short[0].id}.csv").read_text(encoding="utf-8")
    assert ",no" not in sheet and built.routes["weakest"]["reward"] == 0.0


@pytest.mark.parametrize("level", [k for k, v in LEVELS.items() if v.foreign_unit])
def test_the_foreign_unit_sits_on_a_governing_test(level, tmp_path):
    for seed in range(10):
        tests = instance.build(seed, level, tmp_path / str(seed)).scenario.tests
        assert all(t.wet for t in tests if t.crew == "B")


def test_every_level_has_controls_where_defined(tmp_path):
    for level in (k for k, v in LEVELS.items() if v.control):
        controls = [instance.build(s, level, tmp_path / f"{level}-{s}").scenario.control for s in range(40)]
        assert 0 < sum(bool(c) for c in controls) < 40


def test_truth_is_not_in_records(tmp_path):
    built = instance.build(5, 5, tmp_path)
    text = "".join(p.read_text(encoding="utf-8") for p in files(tmp_path))
    assert json.dumps(built.truth) not in text
    assert all(f"{v['fs']:.3f}" not in text for v in built.truth.values())
