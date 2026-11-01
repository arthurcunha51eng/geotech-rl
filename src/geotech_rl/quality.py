"""Instance quality at the level of a whole split: manifests, diversity and the memorisation test."""
import collections
import json
import math
import os
import random
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import grader, instance, scenario, solver

HELDOUT_BASE = 1_000_000    # train and held-out never share a seed
OOD_BASE = 2_000_000
SEED_BASE = {"train": 0, "heldout": HELDOUT_BASE, "ood": OOD_BASE}
MAX_SEEDS_PER_INSTANCE = 10
CONTROL_SHARE_MAX = 0.40    # controls have fewer routes to defeat, so acceptance favours them over the sampled rate


def signature(s: scenario.Scenario) -> str:
    """The discrete shape of an instance: where each defect sits and what it demands."""
    wet = "".join(t.id[3:] for t in s.tests if t.wet)
    cracked = "".join(t.id[3:] for t in s.tests if t.crack_ratio)
    foreign = "".join(f"{t.id[3:]}{t.unit}" for t in s.tests if t.crew == "B")
    faults = ",".join(sorted(f"{p.id}{p.fault}" for p in s.poles if p.fault))
    return f"{s.control}|{wet}|{cracked}|{foreign}|{s.pole_length:g}|{s.fs_required:g}|{faults}"


def _qualify(job: tuple) -> dict | None:
    seed, level, split, draws = job
    scratch = Path(tempfile.mkdtemp(prefix="geotech-q-", dir="/dev/shm" if os.path.isdir("/dev/shm") else None))
    try:
        built = instance.build(seed, level, scratch, split, draws)
        kp = solver.site_constant(scratch)
        dry = solver.site_constant(scratch, "dry_only") if not built.scenario.level.all_wet else None
    except RuntimeError:
        return None
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    s = built.scenario
    wrong = [v for k, v in built.betas.items() if k not in solver.VALID_ROUTES]
    return dict(seed=seed, level=level, split=split, attempt=s.attempt, control=s.control,
                beta_reference=round(min(built.betas[r] for r in solver.VALID_ROUTES), 2),
                beta_wrong=round(min(wrong), 2) if wrong and math.isfinite(min(wrong)) else None,
                phi=round(s.phi, 2), n=round(s.n, 3), c=round(s.c_dry, 2), density=round(s.density, 3),
                length=s.pole_length, required=s.fs_required, kp=round(kp, 4), kp_dry=round(dry, 4) if dry else None, signature=signature(s))


def build_manifest(split: str, level: int, count: int, draws: int, workers: int) -> tuple[list, int]:
    """Qualifies seeds in order until the split holds `count` instances of the level."""
    rows, tried = [], 0
    base = SEED_BASE[split]
    batch = workers * 8
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while len(rows) < count:
            if tried >= count * MAX_SEEDS_PER_INSTANCE + batch:
                raise RuntimeError(f"{split} n{level}: {len(rows)} instances from {tried} seeds; the mechanism is not biting here")
            jobs = [(base + tried + k, level, split, draws) for k in range(batch)]
            rows += [r for r in pool.map(_qualify, jobs, chunksize=4) if r]
            tried += batch
    return rows[:count], tried


def read_manifest(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_manifest(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _entropy(values) -> float:
    counts = np.array(list(collections.Counter(values).values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _shingles(root: Path) -> set:
    text = "".join(p.read_text(encoding="utf-8") for p in sorted(root.rglob("*")) if p.is_file())
    return {text[i:i + 8] for i in range(len(text) - 8)}


def diversity(rows: list, pairs: int = 60) -> dict:
    """Thousand seeds are not a thousand tasks: how many different problems does the split hold?"""
    level = rows[0]["level"]
    wet = [r["signature"].split("|")[1] for r in rows]
    continuous = np.array([[r["phi"], r["n"], r["density"], r["kp"]] for r in rows])
    scaled = (continuous - continuous.mean(0)) / continuous.std(0)
    rng = random.Random(level)
    sample = rng.sample(rows, min(2 * pairs, len(rows)))
    jaccard = []
    with tempfile.TemporaryDirectory() as tmp:
        roots = []
        for i, r in enumerate(sample):
            root = Path(tmp) / str(i)
            instance.build(r["seed"], r["level"], root, r["split"], attempt=r["attempt"])
            roots.append(_shingles(root))
    for a, b in zip(roots[::2], roots[1::2]):
        jaccard.append(len(a & b) / len(a | b))
    nearest = []
    for i in rng.sample(range(len(rows)), min(300, len(rows))):
        d = np.sqrt(((scaled - scaled[i]) ** 2).sum(1))
        d[i] = np.inf
        nearest.append(float(d.min()))
    kp = np.array([r["kp"] for r in rows])
    return dict(
        instances=len(rows), unique_signatures=len({r["signature"] for r in rows}),
        wet_position_entropy_bits=round(_entropy(wet), 2), wet_positions=len(set(wet)),
        controls=dict(collections.Counter(r["control"] or "none" for r in rows)),
        records_jaccard_mean=round(float(np.mean(jaccard)), 3), records_jaccard_max=round(float(np.max(jaccard)), 3),
        nearest_physics_distance_median=round(float(np.median(nearest)), 3),
        site_constant_cv=round(float(kp.std() / kp.mean()), 3))


def _dry_ratios(rows: list) -> np.ndarray:
    return np.array([r["kp_dry"] / r["kp"] for r in rows if r.get("kp_dry")])


def single_factor(train: list, others: dict) -> dict | None:
    """The dry calibration times the one wetting factor that served training best, with no use of the wet test."""
    ratios = _dry_ratios(train)
    if not len(ratios):
        return None
    factors = 1.0 / np.quantile(ratios, np.linspace(0.02, 0.98, 97))
    coverage = [float(np.mean(np.abs(f * ratios - 1.0) <= grader.REL_TOLERANCE)) for f in factors]
    best = float(factors[int(np.argmax(coverage))])
    report = dict(factor=round(best, 3), train=round(max(coverage), 3))
    for name, rows in others.items():
        report[name] = round(float(np.mean(np.abs(best * _dry_ratios(rows) - 1.0) <= grader.REL_TOLERANCE)), 3)
    return report


def null_table(rows: list, sample: int = 60) -> dict:
    """Share of instances each wrong route solves. A route that passes outside the controls marks a mechanism the level lacks."""
    level = rows[0]["level"]
    passed, plain, controls, seen = collections.Counter(), collections.Counter(), 0, 0
    with tempfile.TemporaryDirectory() as tmp:
        for r in rows[:sample]:
            built = instance.build(r["seed"], level, Path(tmp) / str(r["seed"]), r["split"], attempt=r["attempt"])
            seen += 1
            controls += bool(built.scenario.control)
            for route, result in built.routes.items():
                if route not in solver.VALID_ROUTES:
                    passed[route] += result["reward"]
                    plain[route] += 0 if built.scenario.control else result["reward"]
    table = {route: round(passed[route] / seen, 3) for route in sorted(passed)}
    outside = {route: round(plain[route] / max(seen - controls, 1), 3) for route in sorted(plain)}
    declared = instance.mechanisms(scenario.LEVELS[level])
    broken = [route for route in declared if outside.get(route, 0.0) > 0.0]
    share = round(controls / seen, 3)
    warmup = scenario.LEVELS[level].warmup
    return dict(routes=table, outside_controls=outside, declared=list(declared), control_share=share, leaking=broken,
                passed_gate=warmup or (bool(declared) and not broken and share <= CONTROL_SHARE_MAX))


def memorisation(train: list, held: list, sample: int = 100) -> dict:
    """Can an agent pass new sites with the one constant that served training best, without reading the tests?"""
    kp = np.array([r["kp"] for r in train])
    candidates = np.quantile(kp, np.linspace(0.02, 0.98, 97))
    coverage = [float(np.mean(np.abs(c / kp - 1.0) <= grader.REL_TOLERANCE)) for c in candidates]
    constant = float(candidates[int(np.argmax(coverage))])
    passed = {"exact": 0, "memorised": 0}
    chosen = held[:sample]
    with tempfile.TemporaryDirectory() as tmp:
        for r in chosen:
            root = Path(tmp) / f"{r['seed']}"
            built = instance.build(r["seed"], r["level"], root, r["split"], attempt=r["attempt"])
            required = built.scenario.fs_required
            passed["exact"] += grader.grade(built.truth, required, solver.solve(root))["reward"]
            passed["memorised"] += grader.grade(built.truth, required, solver.solve(root, kp=constant))["reward"]
    return dict(train_coverage=round(max(coverage), 3), **{k: round(v / len(chosen), 3) for k, v in passed.items()})
