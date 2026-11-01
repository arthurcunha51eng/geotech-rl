"""Writes a scenario as the site records an inspector would receive. All records are synthetic."""
import csv
import datetime as dt
import io
import math
from pathlib import Path

import numpy as np

from .scenario import NOMINAL_DROP, Scenario

KGF = 0.980665              # daN per kgf
TICKET_BASE = 2200
TICKET_STEP = 7


def _date(s: Scenario, day: int) -> str:
    return (s.start + dt.timedelta(days=day)).isoformat()


def _csv(rows) -> str:
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerows(rows)
    return out.getvalue()


def load_steps(hu: float, rng, stop_ratio: float = 0.0, finish: bool = True):
    """Load steps in the dynamometer unit, coarse first and finer as the pole starts to move."""
    unit = max(20.0, round(hu / 7.0 / 20.0) * 20.0)
    stop = hu * stop_ratio if stop_ratio else hu
    d50 = rng.uniform(16.0, 26.0)
    rows, load = [], 0.0
    while True:
        ratio = load / hu
        load += unit if ratio < 0.7 else unit / 2.0 if ratio < 0.9 else unit / 4.0
        if load > stop:
            if not finish:
                return rows
            final = rows[-1][1] * rng.uniform(2.4, 3.0) if stop_ratio and rows else rng.uniform(480.0, 650.0)
            rows.append((load, round(final, 1), "no"))
            return rows
        f = load / hu
        disp = d50 * f / (1.0 - 0.92 * f) * rng.lognormal(0.0, 0.04)
        rows.append((load, round(disp, 1), "yes"))


def _sheet(s: Scenario, t, rng) -> str:
    factor = {"daN": 1.0, "kgf": 1.0 / KGF, "kN": 0.01}[t.unit]
    steps = load_steps(t.hu_daN, rng, t.short_ratio or t.crack_ratio, finish=not t.short_ratio)
    rows = [("step", f"load_{t.unit}", "disp_mm", "held")]
    for k, (load, disp, held) in enumerate(steps, 1):
        value = load * factor
        rows.append((k, f"{value:.2f}" if t.unit == "kN" else f"{value:.0f}", f"{disp:.1f}", held))
    if not s.level.field:
        return _csv(rows)
    head = (f"# lateral pull test {t.id}  date {_date(s, t.day)}  crew {t.crew}\n"
            f"# test pole: {s.pole_length:g} m precast pole, embedded length {t.L:.2f} m, collar diameter {t.D_sheet:.2f} m\n"
            f"# load applied by cable at {t.h:.2f} m above ground; dynamometer reading in {t.unit}\n")
    return head + _csv(rows) + (f"# remark: {t.remark}\n" if t.remark else "")


def _pull_tests(s: Scenario) -> str:
    """One row per test, for the levels that stay clean. Field levels keep these data inside each sheet."""
    rows = [("test", "date", "crew", "embedded_m", "collar_diam_m", "cable_height_m", "remarks")]
    rows += [(t.id, _date(s, t.day), t.crew, f"{t.L:.2f}", f"{t.D_sheet:.2f}", f"{t.h:.2f}", t.remark) for t in s.tests]
    return _csv(rows)


def _as_built(s: Scenario) -> str:
    rows = [("pole", "nominal_daN", "embedded_m", "collar_diam_m", "installed", "remarks")]
    for p in s.poles:
        pid, depth, diam, remark = p.id, f"{p.L:.2f}", f"{p.D:.2f}", ""
        if p.fault == "comma":
            depth = depth.replace(".", ",")
        elif p.fault == "cm":
            diam, remark = str(int(round(p.D * 100))), "diam in cm"
        elif p.fault == "approx":
            depth, remark = f"~{p.L:.1f}", "depth by foreman, tape not available"
        elif p.fault == "gravel":
            depth, remark = f"{p.L + 0.15:.2f}", "hole depth incl. 0.15 gravel bed under butt"
        elif p.fault == "lowercase":
            pid = pid.lower()
        row = (pid, str(p.rating), depth, diam, _date(s, p.day), remark)
        rows.append(row)
        if p.fault == "duplicate":
            rows.append(row)
    return _csv(rows)


def _compaction(s: Scenario, rng) -> str:
    rows = [(_date(s, t.set_day), t.id, f"{t.dry_density:.2f}", f"{rng.uniform(3.4, 5.6):.1f}") for t in s.tests]
    rows += [(_date(s, p.day), p.id, f"{rng.normal(s.density, 0.018):.2f}", f"{rng.uniform(3.4, 6.0):.1f}")
             for p in s.poles[::3]]
    rows.sort()
    if s.level.typos:
        rows[3] = rows[3][:3] + (f"{float(rows[3][3]) / 100:.3f}",)     # typed as a fraction
        rows[7] = rows[7][:2] + ("", rows[7][3])                        # density not transcribed
    return _csv([("date", "pole", "dry_density_g_cm3", "water_content_pct")] + rows)


def _rain(s: Scenario, rng) -> str:
    rows = [("date", "rain_mm")]
    if not s.level.field:               # compact form for the short episodes
        rows += [(_date(s, d), f"{s.rain[d]:.1f}") for d in sorted(s.rain)]
        return _csv(rows)
    test_days = {t.day for t in s.tests}
    near = {d + k for d in test_days for k in range(-8, 1)}
    free = [d for d in range(s.last_day + 1) if d not in near]
    gaps = set(int(d) for d in rng.choice(free, size=min(3, len(free)), replace=False)) if free else set()
    for day in range(s.last_day + 1):
        rows.append((_date(s, day), "" if day in gaps else f"{s.rain.get(day, 0.0):.1f}"))
    return _csv(rows)


def _concrete(s: Scenario, rng) -> str:
    entries = [(p.day, p.id, p.L, p.D, p.D) for p in s.poles] + [(t.set_day, t.id, t.L, t.D_true, t.D_sheet) for t in s.tests]
    entries.sort(key=lambda e: (e[0], e[1]))
    rows = []
    for k, (day, cid, L, D, auger) in enumerate(entries, 1):
        volume = math.pi / 4.0 * D ** 2 * L * (1.0 + rng.uniform(0.0, 0.06))
        rows.append([_date(s, day), cid, f"{auger * 1000:.0f}", f"{volume:.2f}", f"BT-{TICKET_BASE + TICKET_STEP * k}"])
    if s.swap:
        i = next(k for k, r in enumerate(rows) if r[1] == s.swap[0])
        j = next(k for k, r in enumerate(rows) if r[1] == s.swap[1])
        rows[i][3:5], rows[j][3:5] = rows[j][3:5], rows[i][3:5]
    if s.level.typos:
        wet = next(t.id for t in s.tests if t.wet)
        rest = [r for r in rows if r[1].startswith("L") and r[1] not in (s.swap or ())]
        rest[0][3] = f"{float(rest[0][3]) * 1000:.0f} L"
        rest[1][3], rest[1][4] = "", "ticket lost"
        assert wet not in (rest[0][1], rest[1][1])
    return _csv([["date", "collar", "auger_mm", "concrete_m3", "ticket"]] + rows)


def _lab(s: Scenario, rng) -> str:
    phi = 30.5 + rng.uniform(-1.0, 1.0)
    loads = [50.0, 100.0, 200.0]
    peaks = [q * math.tan(math.radians(phi)) * rng.lognormal(0.0, 0.012) for q in loads]
    fit = math.degrees(math.atan(float(np.dot(loads, peaks) / np.dot(loads, loads))))
    lines = ["SOIL LABORATORY - DIRECT SHEAR TEST REPORT", "Sample: bulk bag, 0.3-1.2 m, fine sand",
             "Specimen remoulded to dry density 1.45 g/cm3, inundated before shearing", "",
             "normal_stress_kPa   peak_shear_kPa"]
    lines += [f"{q:>17.0f}   {t:>14.1f}" for q, t in zip(loads, peaks)]
    lines += ["", f"Strength envelope through the origin: phi = {fit:.1f} deg, c = 0 kPa", ""]
    return "\n".join(lines)


def _readme(s: Scenario) -> str:
    level = s.level
    tests = (["- pull_tests/TP-*.csv: lateral pull tests on test poles set beside the installed ones with the same "
              "procedure. Each row is one load step held for 5 minutes."] if level.field else
             ["- pull_tests.csv: lateral pull tests on test poles set beside the installed ones with the same "
              "procedure, one row per test; cable_height_m is measured above ground.",
              "- pull_tests/TP-*.csv: load steps of each test, each held for 5 minutes. The load column names the "
              "dynamometer unit."])
    files = ["- as_built_poles.csv: installed poles as recorded by the foreman.", *tests,
             "- compaction_control.csv: sand-cone control of the backfill, taken when the poles were set.",
             "- rain_daily.csv: daily rainfall, mm."
             + (" Empty cells are days without a report." if level.field else " Days not listed had no rain.")]
    if level.cave:
        files.append("- concrete_pours.csv: concrete poured into each collar, from the batching tickets, with the "
                     "auger used to dig the hole (mm).")
    if level.lab:
        files.append("- lab_direct_shear.txt: direct shear test on a bulk sample of the site sand.")
    hint = "\nThe backfill counts as soaked after more than 50 mm of rain in the 7 days before a test.\n" if level.hint else "\n"
    if level.method:
        hint += (
            "\nMethod for this site\n"
            "1. gamma (kN/m3) = mean over compaction_control.csv of 9.81 * dry_density * (1 + water_content / 100).\n"
            "2. For each soaked test: H = largest held load in kN, K = H / (0.5 * gamma * D * L**3 / (e + L)), "
            "with D the collar diameter, L the embedded length and e the cable height. Use the mean K.\n"
            f"3. For each pole: e = {s.pole_length:g} - L - {NOMINAL_DROP:.2f}, "
            "H = K * 0.5 * gamma * D * L**3 / (e + L), fs = H * 100 / nominal_daN.\n"
        )
    return (
        "# Lighting pole foundations\n\n"
        "Lengths in m, loads in daN (1 daN = 10 N) unless a sheet says otherwise; 1 kgf = 0.980665 daN; 1 kN = 100 daN.\n"
        f"All poles are {s.pole_length:g} m precast poles in a cast-in-place concrete collar that fills the hole, "
        "flush with the ground, in fine sand compacted in layers.\n"
        f"Nominal load of a pole: horizontal load equal to its rating, applied {NOMINAL_DROP:.2f} m below the top.\n"
        f"Required safety factor: {s.fs_required:g}.\n\n"
        "Files\n" + "\n".join(files) + "\n\n"
        "Conventions\n"
        "- Ultimate lateral resistance in a pull test: the largest load held for the full step, at that test's cable height.\n"
        + ("- A collar that cracks during a test ends the test; its loads do not measure the soil.\n"
           "- The foundations must hold through the rainy season, when the backfill is soaked.\n" if level.hint else "")
        + hint
    )


def instruction(s: Scenario, tools: bool = False) -> str:
    where = "in the working directory" if tools else "in /app/data"
    deliver = "Call submit with a JSON object" if tools else "Write /app/output/pole_check.json as a JSON object"
    return (
        f"Decide which of the {s.level.n_poles} lighting poles must have their foundations redone. "
        f"The site records are {where}; README.md gives the conventions.\n\n"
        "For every installed pole, compute the safety factor of its foundation: ultimate lateral resistance for a "
        "horizontal load at the nominal-load point, divided by the nominal load. It must hold throughout the service "
        f"life of the poles. A pole must be re-founded when its safety factor is below {s.fs_required:g}. "
        "Give best estimates, not conservative values.\n\n"
        f'{deliver} keyed by pole ID, each entry with a numeric "fs" and a boolean "refound". '
        'Each "fs" must be within 10% of the actual safety factor. Each "refound" must agree with '
        "the actual safety factor, except that a pole within 10% of the requirement may be flagged either way.\n"
    )


def write(s: Scenario, data_dir: Path, draw: int = 0) -> None:
    """Writes every record of the scenario; the seed stream below fixes all clerical detail."""
    rng = np.random.default_rng([s.seed, s.level.id, s.attempt, 99, draw])
    (data_dir / "pull_tests").mkdir(parents=True, exist_ok=True)
    (data_dir / "README.md").write_text(_readme(s), encoding="utf-8")
    (data_dir / "as_built_poles.csv").write_text(_as_built(s), encoding="utf-8")
    (data_dir / "compaction_control.csv").write_text(_compaction(s, rng), encoding="utf-8")
    (data_dir / "rain_daily.csv").write_text(_rain(s, rng), encoding="utf-8")
    if not s.level.field:
        (data_dir / "pull_tests.csv").write_text(_pull_tests(s), encoding="utf-8")
    for t in s.tests:
        (data_dir / "pull_tests" / f"{t.id}.csv").write_text(_sheet(s, t, rng), encoding="utf-8")
    if s.level.cave:
        (data_dir / "concrete_pours.csv").write_text(_concrete(s, rng), encoding="utf-8")
    if s.level.lab:
        (data_dir / "lab_direct_shear.txt").write_text(_lab(s, rng), encoding="utf-8")
