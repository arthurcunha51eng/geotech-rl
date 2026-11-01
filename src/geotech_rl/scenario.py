"""Samples one pole-foundation scenario from (seed, level). Nothing here touches the file system."""
import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from . import physics

NOMINAL_DROP = 0.10         # nominal load point below the pole top, m
CREW_A_DROP = 0.30          # crew A hangs the cable this far below the top, m
SIZES = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00)
RATINGS = (50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1500, 2000)
LD_RANGE = (3.3, 4.5)       # embedment / collar diameter of a foundation table
WATER_DRY = (0.031, 0.048)
WATER_WET = (0.162, 0.184)
SUCTION_TRAP = (3.5, 10.0)  # kPa: the dry-test route fails every pole
SUCTION_CONTROL = (1.0, 2.0)  # kPa: the dry-test route is still correct
SUCTION_OOD = (10.0, 18.0)  # kPa: friction is so high out there that less suction would make dry and soaked tests agree
CONTROL_RATE = 0.20
SCATTER = 0.03              # lognormal scatter of a pull test
SHORT_OF_SOAKED = (0.55, 0.75)  # where a test stopped early ends, over what the soaked soil would have held
AS_BUILT_FAULTS = ("comma", "cm", "approx", "gravel", "lowercase", "duplicate")

# train and heldout share one distribution and differ in seeds; ood sits outside every training range.
# n is log-uniform and wide on purpose: a narrow site constant could be memorised instead of measured.
SPLITS = {
    "train": dict(phi=(28.0, 38.0), n=(0.8, 3.2), density=(1.55, 1.70), lengths=(7.0, 9.0, 11.0), required=(1.5, 2.0, 2.5),
                  suction=SUCTION_TRAP),
    "ood": dict(phi=(38.0, 41.0), n=(3.2, 4.0), density=(1.70, 1.78), lengths=(13.0,), required=(3.0,),
                suction=SUCTION_OOD),
}
SPLITS["heldout"] = SPLITS["train"]


@dataclass(frozen=True)
class Level:
    id: int
    n_tests: int
    n_poles: int
    all_wet: bool
    crack: bool
    foreign_unit: bool
    cave: bool
    lab: bool
    faults: int
    hint: bool
    near_limit: bool
    control: tuple[str, ...]
    method: bool = False
    wet_tests: int = 1      # soaked tests when the others are dry: the traps sit on these, or they change nothing
    field: bool = False     # field sheets: the test data sit in comment lines, the remark under the table
    typos: bool = False     # transcription slips in the control and pour records
    warmup: bool = False    # no wrong route fails here: it trains format and routine, and never counts as a task


# N0-N1 warm up. N2-N4 add one mechanism each, with its rule written in the records. N5 joins the three with no rule
# written. N6-N8 add the field: dirty records, the collar found by concrete volume, and the decoys.
LEVELS = {
    0: Level(0, 2, 3, True, False, False, False, False, 0, True, False, (), method=True, warmup=True),
    1: Level(1, 4, 6, True, False, False, False, False, 0, True, False, (), warmup=True),
    2: Level(2, 4, 6, True, False, True, False, False, 0, True, False, ("soft_unit",)),
    3: Level(3, 3, 6, True, True, False, False, False, 0, True, False, ("hairline",)),
    4: Level(4, 5, 6, False, False, False, False, False, 0, True, False, ("low_suction",), wet_tests=2),
    5: Level(5, 6, 8, False, True, True, False, False, 0, False, False, ("soft_unit", "hairline", "low_suction"), wet_tests=3),
    6: Level(6, 6, 8, False, True, True, False, False, 2, False, False, ("soft_unit", "hairline", "low_suction"), wet_tests=3,
             field=True),
    7: Level(7, 6, 10, False, True, True, True, False, 3, False, False, ("hairline", "ticket_swap"), wet_tests=3, field=True),
    8: Level(8, 8, 22, False, True, True, True, True, 6, False, True, ("low_suction", "ticket_swap"), wet_tests=3, field=True,
             typos=True),
}


@dataclass
class Pole:
    id: str
    rating: int
    L: float
    D: float
    day: int
    fs: float = 0.0
    fault: str = ""


@dataclass
class PullTest:
    id: str
    L: float
    D_sheet: float
    D_true: float
    crew: str
    day: int
    set_day: int
    wet: bool
    h: float
    unit: str
    dry_density: float
    water: float
    hu_daN: float
    hu_model_daN: float = 0.0
    remark: str = ""
    crack_ratio: float = 0.0
    short_ratio: float = 0.0    # stopped before failure: its largest held load says nothing about the soil


@dataclass
class Scenario:
    seed: int
    level: Level
    split: str
    attempt: int
    control: str
    phi: float
    n: float
    c_dry: float
    density: float
    gamma_wet: float
    pole_length: float
    fs_required: float
    start: dt.date
    poles: list[Pole] = field(default_factory=list)
    tests: list[PullTest] = field(default_factory=list)
    rain: dict[int, float] = field(default_factory=dict)
    last_day: int = 0
    swap: tuple[str, str] | None = None


def _geometry(rng, scale, lo, hi):
    L = round(rng.uniform(lo * scale, hi * scale) / 0.05) * 0.05
    ratio = rng.uniform(*LD_RANGE)
    return round(L, 2), min(SIZES, key=lambda d: abs(L / d - ratio))


def _pole_capacity(s: Scenario, L, D):
    e = s.pole_length - L - NOMINAL_DROP
    return physics.ultimate_load(L, D, e, s.phi, s.gamma_wet, 0.0, s.n) * 100.0


def _sample_poles(s: Scenario, rng, scale):
    """Installed poles with true safety factors spread around the requirement."""
    level = s.level
    ids = [f"L-{i:02d}" for i in range(1, level.n_poles + 1)]
    big = rng.random(level.n_poles) < 0.65
    geometry = [_geometry(rng, scale, *((1.80, 2.65) if b else (1.35, 1.95))) for b in big]
    capacity = np.array([_pole_capacity(s, L, D) for L, D in geometry])

    def rating_for(mask):
        target = np.median(capacity[mask]) / s.fs_required if mask.any() else RATINGS[0]
        return min(RATINGS, key=lambda r: abs(r - target))

    rating_big, rating_small = rating_for(big), rating_for(~big)
    faults = list(rng.permutation(AS_BUILT_FAULTS))[:level.faults]
    picks = list(rng.permutation(ids))
    fault_of = dict(zip(picks, faults))
    for i, pid in enumerate(ids):
        L, D = geometry[i]
        if fault_of.get(pid) == "approx":
            L = round(L, 1)
        pole = Pole(pid, rating_big if big[i] else rating_small, L, D, int(rng.integers(0, 21)),
                    fault=fault_of.get(pid, ""))
        pole.fs = _pole_capacity(s, pole.L, pole.D) / pole.rating
        s.poles.append(pole)


def _test_days(s: Scenario, rng):
    """Places the storms and returns one (day, wet) per test."""
    level = s.level
    if level.all_wet:
        storms = [int(rng.integers(30, 40)), int(rng.integers(55, 65))]
        days = [(int(storms[i % 2] + 2 + rng.integers(0, 3)), True) for i in range(level.n_tests)]
    else:
        storm = int(rng.integers(45, 60))
        storms = [storm]
        wet = [int(d) for d in rng.choice(list(range(storm + 2, storm + 7)), size=level.wet_tests, replace=False)]
        dry = [int(d) for d in rng.choice(
            list(range(30, storm)) + list(range(storm + 10, 81)), size=level.n_tests - level.wet_tests, replace=False)]
        days = [(d, True) for d in wet] + [(d, False) for d in dry]
    for storm in storms:
        total = rng.uniform(60.0, 110.0)
        split = rng.uniform(0.4, 0.7)
        s.rain[storm] = round(total * split, 1)
        s.rain[storm + 1] = round(total * (1.0 - split), 1)
    s.last_day = max(d for d, _ in days) + 3
    for day in range(s.last_day + 1):
        if day not in s.rain and rng.random() < 0.15:
            s.rain[day] = round(rng.uniform(0.3, 5.0), 1)
    rng.shuffle(days)
    return days


def _sample_tests(s: Scenario, rng, scale):
    level = s.level
    days = _test_days(s, rng)
    wet_ids = [i for i, (_, wet) in enumerate(days) if wet]
    target = wet_ids[0]
    caved = level.cave and s.control != "ticket_swap" and s.control != "low_suction"
    traps = [int(i) for i in rng.permutation(wet_ids[1:])] if level.crack or level.foreign_unit else []
    cracked = traps.pop() if level.crack else None
    foreign = traps.pop() if level.foreign_unit else None
    # Soaked tests are the weakest, so "take the lowest" would pass without reading the rain. This dry test ends lower.
    short = int(rng.choice([i for i, (_, wet) in enumerate(days) if not wet])) if not level.all_wet else None
    for i, (day, wet) in enumerate(days):
        tid = f"TP-{i + 1}"
        if i == target and level.cave:
            L, D = _geometry(rng, scale, 1.95, 2.25)
        else:
            L, D = _geometry(rng, scale, 1.50, 2.20)
        D_true = D
        if i == target and caved:
            D_true = round(D * rng.uniform(1.25, 1.40), 3)
        crew = "B" if i == foreign else "A"
        h = round(s.pole_length - L - CREW_A_DROP, 2) if crew == "A" else round(rng.uniform(0.60, 0.72) * s.pole_length, 2)
        unit = "daN" if crew == "A" else "kgf" if s.control == "soft_unit" else "kN"
        density = round(float(rng.normal(s.density, 0.018)), 2)
        water = float(rng.uniform(*(WATER_WET if wet else WATER_DRY)))
        cohesion = 0.0 if wet else s.c_dry
        gamma = physics.unit_weight(density, water)
        model = physics.ultimate_load(L, D_true, h, s.phi, gamma, cohesion, s.n) * 100.0
        test = PullTest(tid, L, D, D_true, crew, day, int(rng.integers(0, 21)), wet, h, unit, density, water,
                        float(model * rng.lognormal(0.0, SCATTER)), float(model))
        if i == short:
            soaked = physics.ultimate_load(L, D_true, h, s.phi, gamma, 0.0, s.n) * 100.0
            test.short_ratio = float(soaked * rng.uniform(*SHORT_OF_SOAKED) / test.hu_daN)
        if i == cracked:
            if s.control == "hairline":
                test.remark = "hairline crack at the collar seen after unloading, none during the test"
            else:
                test.crack_ratio = float(rng.uniform(0.45, 0.60))
                test.remark = "collar cracked at ground line during the last step; test stopped"
        s.tests.append(test)


def _swap_tickets(s: Scenario, rng):
    """Books the wet test's ticket against a collar of very different volume, poured on the same day."""
    wet = next(t for t in s.tests if t.wet)
    volume = wet.D_true ** 2 * wet.L
    far = [p for p in s.poles if not 0.55 < (p.D ** 2 * p.L) / volume < 1.8]
    partner = far[int(rng.integers(len(far)))] if far else max(s.poles, key=lambda p: abs(p.D ** 2 * p.L - volume))
    partner.day = wet.set_day
    s.swap = (wet.id, partner.id)


def sample(seed: int, level_id: int, attempt: int = 0, split: str = "train") -> Scenario:
    level = LEVELS[level_id]
    ranges = SPLITS[split]
    rng = np.random.default_rng([seed, level_id, attempt, list(SPLITS).index(split)])
    control = ""
    if level.control:       # drawn from the seed alone: rejected attempts must not raise the share of the easy instances
        choice = np.random.default_rng([seed, level_id, list(SPLITS).index(split), 31])
        control = str(choice.choice(level.control)) if choice.random() < CONTROL_RATE else ""
    pole_length = float(rng.choice(ranges["lengths"]))
    density = float(rng.uniform(*ranges["density"]))
    c_range = SUCTION_CONTROL if control == "low_suction" else ranges["suction"]
    s = Scenario(
        seed=seed, level=level, split=split, attempt=attempt, control=control,
        phi=float(rng.uniform(*ranges["phi"])), n=float(np.exp(rng.uniform(*np.log(ranges["n"])))),
        c_dry=float(rng.uniform(*c_range)),
        density=density, gamma_wet=physics.unit_weight(density, sum(WATER_WET) / 2.0),
        pole_length=pole_length, fs_required=float(rng.choice(ranges["required"])),
        start=dt.date(int(rng.integers(2019, 2026)), int(rng.integers(1, 13)), 1),
    )
    scale = (pole_length / 9.0) ** 0.8
    _sample_poles(s, rng, scale)
    _sample_tests(s, rng, scale)
    if control == "ticket_swap":
        _swap_tickets(s, rng)
    elif level.typos:                   # a booking error on installed poles: looks like a defect, changes nothing
        first, second = rng.choice(len(s.poles), size=2, replace=False)
        s.poles[second].day = s.poles[first].day
        s.swap = (s.poles[first].id, s.poles[second].id)
    return s


def redraw_scatter(s: Scenario, rng) -> None:
    """Same site and same collars; only the pull-test scatter changes."""
    for t in s.tests:
        t.hu_daN = t.hu_model_daN * float(rng.lognormal(0.0, SCATTER))


def truth(s: Scenario) -> dict:
    return {p.id: {"fs": round(p.fs, 4), "refound": bool(p.fs < s.fs_required)} for p in s.poles}
