"""Gate before any paid run: a scripted agent must solve real instances through the agent's own sandbox.

Unit tests prove pieces; this proves the path a model actually takes. It writes code the way a model does, with
pandas and the usual standard library, and runs it through sandbox.run in the environment that will serve training.

    python preflight.py
"""
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from geotech_rl import grader, instance, sandbox, scenario

LEVELS = (0, 1)
SEEDS = 5

LIBRARIES = """
import collections, csv, datetime, functools, glob, itertools, json, math, os, pathlib, random, re, statistics, string
import numpy as np, pandas as pd, scipy
from scipy import integrate, interpolate, optimize, stats
print("LIBRARIES_OK")
"""

# Default arguments on purpose: a first attempt by a model. Messy values are the task; an unreadable file is not.
PLAIN_READ = """
import glob
import pandas as pd
for path in glob.glob("*.csv") + glob.glob("pull_tests/*.csv"):
    assert len(pd.read_csv(path).columns) >= 2, path
print("PLAIN_READ_OK")
"""

AGENT = r"""
import json, re
import pandas as pd

readme = open("README.md").read()
length = float(re.search(r"All poles are ([\d.]+) m", readme).group(1))
required = float(re.search(r"Required safety factor: (\d+(?:\.\d+)?)", readme).group(1))
control = pd.read_csv("compaction_control.csv")
gamma = (9.81 * control["dry_density_g_cm3"] * (1 + control["water_content_pct"] / 100)).mean()
constants = []
for _, test in pd.read_csv("pull_tests.csv").iterrows():
    sheet = pd.read_csv(f"pull_tests/{test['test']}.csv")
    held = sheet.loc[sheet["held"] == "yes", "load_daN"].max() / 100
    L, D, h = test["embedded_m"], test["collar_diam_m"], test["cable_height_m"]
    constants.append(held / (0.5 * gamma * D * L ** 3 / (h + L)))
K = sum(constants) / len(constants)
answer = {}
for _, pole in pd.read_csv("as_built_poles.csv").iterrows():
    L, D = float(pole["embedded_m"]), float(pole["collar_diam_m"])
    e = length - L - 0.10
    fs = K * 0.5 * gamma * D * L ** 3 / (e + L) * 100 / float(pole["nominal_daN"])
    answer[pole["pole"]] = {"fs": round(float(fs), 3), "refound": bool(fs < required)}
print(json.dumps(answer))
"""


def check(condition: bool, message: str) -> None:
    if not condition:
        sys.exit(f"PREFLIGHT_FAILED: {message}")


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output = await sandbox.run(LIBRARIES, Path(tmp))
        check("LIBRARIES_OK" in output, f"libraries blocked\n{output}")
    for level in (k for k, v in scenario.LEVELS.items() if not v.field):
        with tempfile.TemporaryDirectory() as tmp:
            instance.build(0, level, Path(tmp))
            output = await sandbox.run(PLAIN_READ, Path(tmp))
        check("PLAIN_READ_OK" in output, f"a record of N{level} fails a plain read\n{output}")
    solved = 0
    for level in LEVELS:
        for seed in range(SEEDS):
            with tempfile.TemporaryDirectory() as tmp:
                built = instance.build(seed, level, Path(tmp))
                output = await sandbox.run(AGENT, Path(tmp))
            check("blocked in this sandbox" not in output, f"sandbox blocked agent code\n{output}")
            check(output.strip().startswith("{"), f"agent code failed on N{level} seed {seed}\n{output}")
            result = grader.grade(built.truth, built.scenario.fs_required, json.loads(output))
            check(result["reward"] == 1.0, f"scripted agent scored {result} on N{level} seed {seed}")
            solved += 1
    print(f"PREFLIGHT_OK {solved} episodes solved through the sandbox")


if __name__ == "__main__":
    asyncio.run(main())
