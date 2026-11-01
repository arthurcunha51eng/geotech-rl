"""Builds accepted instances: the reference passes, the mechanism bites, and both hold under test scatter."""
import copy
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import grader, records, scenario, solver

MAX_ATTEMPTS = 400
FLAG_SHARE = (0.25, 0.75)   # share of poles to re-found
MIN_SPREAD = 1.6            # strongest over weakest pole: constant answers must fail
BETA_MIN = 2.0              # margin to the tolerance, in standard deviations of the scatter
SIGMA_FLOOR = 1e-3
NOISELESS = ("constant", "geometry_index", "lab_phi")   # these routes never read a pull test


@dataclass
class Instance:
    scenario: scenario.Scenario
    truth: dict
    routes: dict
    betas: dict = field(default_factory=dict)


def mechanisms(level: scenario.Level) -> tuple[str, ...]:
    """Wrong routes the level exists to defeat. A control instance lifts one of them, never all of them."""
    routes = []
    if level.foreign_unit:
        routes.append("ignore_units")
    if level.crack:
        routes.append("keep_cracked")
    if not level.all_wet:
        routes += ["dry_only", "all_tests", "weakest", "latest"]
    if level.cave:
        routes.append("sheet_diameter")
    if level.lab:
        routes.append("lab_phi")
    return tuple(routes)


def must_fail(s: scenario.Scenario) -> tuple[str, ...]:
    """Wrong routes the scenario has to defeat, or its mechanism would not be tested."""
    routes = ["constant", "geometry_index"]
    if s.level.foreign_unit and s.control != "soft_unit":
        routes.append("ignore_units")
    if s.level.crack and s.control != "hairline":
        routes.append("keep_cracked")
    if not s.level.all_wet:
        routes.append("weakest")
    if not s.level.all_wet and s.control != "low_suction":
        routes += ["dry_only", "all_tests", "latest"]
    if s.control == "ticket_swap":
        routes.append("poured_diameter")
    elif s.level.cave and s.control == "":
        routes.append("sheet_diameter")
    if s.level.lab:
        routes.append("lab_phi")
    return tuple(routes)


def _table_ok(s: scenario.Scenario) -> bool:
    fs = np.array([p.fs for p in s.poles])
    share = float(np.mean(fs < s.fs_required))
    distance = np.abs(fs - s.fs_required) / s.fs_required
    if not FLAG_SHARE[0] <= share <= FLAG_SHARE[1] or fs.max() / fs.min() < MIN_SPREAD or distance.min() < 0.01:
        return False
    if s.level.near_limit:
        return bool(np.any((fs < s.fs_required) & (distance <= 0.10)) and np.any((fs > s.fs_required) & (distance <= 0.10)))
    return True


def _scratch() -> Path:
    base = "/dev/shm" if os.path.isdir("/dev/shm") else None
    return Path(tempfile.mkdtemp(prefix="geotech-", dir=base))


def evaluate(s: scenario.Scenario, data: Path) -> tuple[dict, dict]:
    truth = scenario.truth(s)
    # valid routes must never fail silently
    routes = {r: grader.grade(truth, s.fs_required, solver.solve(data, r)) for r in solver.VALID_ROUTES}
    for route in solver.ROUTES[len(solver.VALID_ROUTES):]:
        if route == "lab_phi" and not s.level.lab:
            continue
        try:
            routes[route] = grader.grade(truth, s.fs_required, solver.solve(data, route))
        except (ValueError, ZeroDivisionError):        # a wrong route that cannot run on this level
            routes[route] = dict(reward=0.0, partial=0.0, decision_ok=0.0)
    return truth, routes


def _accepted(s: scenario.Scenario, routes: dict) -> bool:
    valid = all(routes[r]["reward"] == 1.0 for r in solver.VALID_ROUTES)
    return valid and all(routes[r]["reward"] == 0.0 for r in must_fail(s))


def diagnose(data: Path, answer) -> str:
    """The route an answer followed, so a failure reads as a mechanism and not just as a zero."""
    errors = {}
    for route in solver.ROUTES:
        try:
            errors[route] = grader.worst_error(solver.solve(data, route), answer)
        except (ValueError, ZeroDivisionError, FileNotFoundError):
            continue                                    # a route this level's records cannot run
    route = min(errors, key=errors.get)
    return route if errors[route] <= grader.REL_TOLERANCE else "other"


def reliability(s: scenario.Scenario, draws: int, scratch: Path) -> dict:
    """Reliability index of each route against the 10% tolerance, over fresh draws of the test scatter.

    beta_reference: (tolerance - mean worst error) / std, the margin by which the right route passes.
    beta_<route>: (mean worst error - tolerance) / std, the margin by which a wrong route fails.
    """
    routes = solver.VALID_ROUTES + tuple(r for r in must_fail(s) if r not in NOISELESS)
    errors = {r: [] for r in routes}
    trial = copy.deepcopy(s)
    truth = scenario.truth(s)
    rng = np.random.default_rng([s.seed, s.level.id, s.attempt, 7])
    for draw in range(1, draws + 1):
        scenario.redraw_scatter(trial, rng)
        records.write(trial, scratch, draw)
        for route in routes:
            try:
                errors[route].append(grader.worst_error(truth, solver.solve(scratch, route)))
            except (ValueError, ZeroDivisionError):
                errors[route].append(math.inf)
    betas = {}
    for route, values in errors.items():
        finite = np.array([v for v in values if math.isfinite(v)])
        if len(finite) < len(values) or len(finite) == 0:
            betas[route] = -math.inf if route in solver.VALID_ROUTES else math.inf
            continue
        valid = route in solver.VALID_ROUTES
        margin = grader.REL_TOLERANCE - finite.mean() if valid else finite.mean() - grader.REL_TOLERANCE
        betas[route] = float(margin / max(finite.std(), SIGMA_FLOOR))
    return betas


def build(seed: int, level: int, data_dir: Path, split: str = "train", draws: int = 0, attempt: int | None = None) -> Instance:
    """First accepted attempt of (seed, level), written to data_dir; a known attempt skips the search."""
    attempts = range(MAX_ATTEMPTS) if attempt is None else (attempt,)
    for k in attempts:
        s = scenario.sample(seed, level, k, split)
        if not _table_ok(s):
            continue
        scratch = _scratch()
        try:
            records.write(s, scratch)
            truth, routes = evaluate(s, scratch)
            if not _accepted(s, routes):
                continue
            betas = reliability(s, draws, scratch) if draws else {}
            if betas and min(betas.values()) < BETA_MIN:
                continue
            records.write(s, Path(data_dir))
            return Instance(s, truth, routes, betas)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    raise RuntimeError(f"no accepted instance for seed {seed}, level {level}, split {split}")
