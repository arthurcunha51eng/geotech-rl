"""Reference solution and the plausible wrong routes. Reads only the site records, never the scenario."""
import csv
import datetime as dt
import math
import re
from pathlib import Path

import numpy as np

from .physics import G, broms_unit, passive_coefficient, rotation_unit
from .records import KGF
from .scenario import NOMINAL_DROP

WET_RAIN_MM = 50.0
DIAMETER_TOLERANCE = 0.15
VOLUME_MATCH = 0.10
TO_DAN = {"daN": 1.0, "kgf": KGF, "kN": 100.0}

ROUTES = ("reference", "rotation", "mid_step", "dry_only", "all_tests", "keep_cracked", "ignore_units", "sheet_diameter",
          "poured_diameter", "lab_phi", "constant", "geometry_index", "weakest", "latest")
VALID_ROUTES = ("reference", "rotation", "mid_step")     # different engineering choices that must all pass


def read_project(data: Path) -> dict:
    text = (data / "README.md").read_text(encoding="utf-8")
    return dict(length=float(re.search(r"All poles are (\d+(?:\.\d+)?) m", text).group(1)),
                fs_required=float(re.search(r"Required safety factor: (\d+(?:\.\d+)?)", text).group(1)))


def read_steps(lines: list[str]) -> dict:
    steps = list(csv.DictReader(line for line in lines if not line.startswith("#")))
    column = next(name for name in steps[0] if name.startswith("load_"))
    held = [float(step[column]) for step in steps if step["held"].strip() == "yes"]
    failed = [float(step[column]) for step in steps if step["held"].strip() != "yes"]
    return dict(unit=column.removeprefix("load_"), held=max(held, default=0.0), failed=min(failed, default=0.0))


def read_test(row: dict, path: Path) -> dict:
    return dict(id=row["test"], L=float(row["embedded_m"]), D=float(row["collar_diam_m"]), h=float(row["cable_height_m"]),
                date=row["date"], cracked="cracked at ground line during" in row["remarks"],
                **read_steps(path.read_text(encoding="utf-8").splitlines()))


def read_field_sheet(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    head = "\n".join(line for line in lines if line.startswith("#"))
    return dict(id=path.stem, L=float(re.search(r"embedded length ([\d.]+) m", head).group(1)),
                D=float(re.search(r"collar diameter ([\d.]+) m", head).group(1)),
                h=float(re.search(r"cable at ([\d.]+) m above ground", head).group(1)),
                date=re.search(r"date (\S+)", head).group(1), cracked="cracked at ground line during" in head,
                **read_steps(lines))


def read_tests(data: Path) -> list:
    index = data / "pull_tests.csv"
    if not index.exists():
        return [read_field_sheet(path) for path in sorted((data / "pull_tests").glob("*.csv"))]
    with index.open(encoding="utf-8") as f:
        return [read_test(row, data / "pull_tests" / f"{row['test']}.csv") for row in csv.DictReader(f)]


def read_poles(path: Path) -> dict:
    poles = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["pole"].strip().upper()
            L = float(row["embedded_m"].replace("~", "").replace(",", "."))
            gravel = re.search(r"incl\. ([\d.]+) gravel", row["remarks"])
            if gravel:
                L -= float(gravel.group(1))
            D = float(row["collar_diam_m"].replace(",", "."))
            if D > 5.0:                                 # recorded in cm
                D /= 100.0
            poles[pid] = dict(rating=float(row["nominal_daN"]), L=L, D=D)
    return poles


def read_unit_weight(path: Path) -> float:
    gammas = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["dry_density_g_cm3"] or not row["water_content_pct"]:
                continue
            w = float(row["water_content_pct"])
            w = w / 100.0 if w > 1.0 else w             # a few entries are typed as fractions
            gammas.append(float(row["dry_density_g_cm3"]) * (1.0 + w) * G)
    return float(np.mean(gammas))


def rain_before(path: Path, date: str, days: int = 7) -> float:
    last = dt.date.fromisoformat(date)
    total = 0.0
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = dt.date.fromisoformat(row["date"])
            if last - dt.timedelta(days=days) <= d <= last and row["rain_mm"]:
                total += float(row["rain_mm"])
    return total


def read_pours(path: Path) -> list:
    pours = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cell = row["concrete_m3"].strip()
            volume = None
            if cell:
                volume = float(cell[:-1]) / 1000.0 if cell.endswith("L") else float(cell)
            pours.append(dict(collar=row["collar"].strip().upper(), date=row["date"], volume=volume))
    return pours


def _volume(geometry: dict) -> float:
    return math.pi / 4.0 * geometry["D"] ** 2 * geometry["L"]


def _swapped(test: dict, pours: list, geometry: dict) -> bool:
    """A ticket that fits a collar of the same day, whose own ticket fits this one, is a booking error."""
    mine = next(p for p in pours if p["collar"] == test["id"])
    for other in pours:
        if other is mine or other["date"] != mine["date"] or other["volume"] is None or other["collar"] not in geometry:
            continue
        fits_there = abs(mine["volume"] / _volume(geometry[other["collar"]]) - 1.0) < VOLUME_MATCH
        fits_here = abs(other["volume"] / _volume(test) - 1.0) < VOLUME_MATCH
        if fits_there and fits_here:
            return True
    return False


def _correct_diameters(tests: list, poles: dict, data: Path, route: str) -> None:
    path = data / "concrete_pours.csv"
    if not path.exists() or route == "sheet_diameter":
        return
    pours = read_pours(path)
    volumes = {p["collar"]: p["volume"] for p in pours if p["volume"] is not None}
    geometry = {**poles, **{t["id"]: t for t in tests}}
    for t in tests:
        if t["id"] not in volumes:
            continue
        poured = math.sqrt(4.0 * volumes[t["id"]] / (math.pi * t["L"]))
        if abs(poured / t["D"] - 1.0) <= DIAMETER_TOLERANCE:
            continue
        if route == "poured_diameter" or not _swapped(t, pours, geometry):
            t["D"] = poured


def calibrate(data: Path, poles: dict, gamma: float, route: str = "reference") -> float:
    """Site constant from the pull tests. One model and one unit weight serve calibration and prediction,
    so kp is a constant of the method on this site, not a soil property."""
    tests = read_tests(data)
    for t in tests:
        load = 0.5 * (t["held"] + t["failed"]) if route == "mid_step" and t["failed"] else t["held"]
        t["hu"] = load * (1.0 if route == "ignore_units" else TO_DAN[t["unit"]]) / 100.0
        t["wet"] = rain_before(data / "rain_daily.csv", t["date"]) > WET_RAIN_MM
    _correct_diameters(tests, poles, data, route)
    valid = tests if route == "keep_cracked" else [t for t in tests if not t["cracked"]]
    if route in ("all_tests", "weakest", "latest"):
        chosen = valid
    elif route == "dry_only":
        chosen = [t for t in valid if not t["wet"]] or valid
    else:
        chosen = [t for t in valid if t["wet"]]
    unit = rotation_unit if route == "rotation" else broms_unit
    constants = {t["date"]: t["hu"] / unit(t["L"], t["D"], t["h"], gamma) for t in chosen}
    if route == "weakest":
        return float(min(constants.values()))
    if route == "latest":
        return float(constants[max(constants)])
    return float(np.mean([t["hu"] / unit(t["L"], t["D"], t["h"], gamma) for t in chosen]))


def site_constant(data: Path, route: str = "reference") -> float:
    data = Path(data)
    return calibrate(data, read_poles(data / "as_built_poles.csv"), read_unit_weight(data / "compaction_control.csv"), route)


def solve(data: Path, route: str = "reference", kp: float | None = None) -> dict:
    """Safety factor and decision for every installed pole. A given kp stands for a memorised site constant."""
    data = Path(data)
    project = read_project(data)
    poles = read_poles(data / "as_built_poles.csv")
    required = project["fs_required"]
    length = project["length"]
    lever = {k: length - p["L"] - NOMINAL_DROP for k, p in poles.items()}

    def answer(fs):
        return {k: {"fs": round(float(v), 3), "refound": bool(v < required)} for k, v in fs.items()}

    if route == "constant":
        return answer({k: required for k in poles})
    if route == "geometry_index":
        index = {k: p["D"] * p["L"] ** 3 / (lever[k] + p["L"]) / p["rating"] for k, p in poles.items()}
        median = float(np.median(list(index.values())))
        return answer({k: required * v / median for k, v in index.items()})

    gamma = read_unit_weight(data / "compaction_control.csv")
    if kp is None and route == "lab_phi":
        text = (data / "lab_direct_shear.txt").read_text(encoding="utf-8")
        kp = passive_coefficient(float(re.search(r"phi = ([\d.]+)", text).group(1)))
    elif kp is None:
        kp = calibrate(data, poles, gamma, route)
    unit = rotation_unit if route == "rotation" else broms_unit
    fs = {k: kp * unit(p["L"], p["D"], lever[k], gamma) * 100.0 / p["rating"] for k, p in poles.items()}
    return answer(fs)
