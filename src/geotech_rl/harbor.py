"""Exports an instance as a Harbor task: separate verifier, truth outside the agent's container, no network."""
import json
import shutil
from pathlib import Path

from . import records
from .instance import Instance, build

TASK_TOML = """schema_version = "1.4"
artifacts = ["/app/output/pole_check.json"]

[task]
name = "geotech-rl/{name}"
description = "Decide which pole foundations must be redone from site records (synthetic, level {level})."

[verifier]
timeout_sec = 120.0
environment_mode = "separate"

[verifier.environment]
network_mode = "no-network"

[agent]
timeout_sec = 1800.0

[environment]
build_timeout_sec = 900.0
cpus = 2
memory_mb = 2048
storage_mb = 4096
gpus = 0
network_mode = "no-network"
"""

ENV_DOCKERFILE = """FROM python:3.12-slim-bookworm
RUN pip install --no-cache-dir numpy==2.2.6 scipy==1.15.3
WORKDIR /app
COPY data /app/data
RUN mkdir -p /app/output
"""

TEST_DOCKERFILE = """FROM python:3.12-slim-bookworm
COPY . /tests/
RUN mkdir -p /app/output /logs/verifier && chmod +x /tests/test.sh
"""

TEST_SH = """#!/bin/bash
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
cd /tests || exit 0
python3 /tests/verify.py || true
exit 0
"""

VERIFY_PY = """import json
from pathlib import Path

import grader

answer_path = Path("/app/output/pole_check.json")
expected = json.loads(Path("/tests/truth.json").read_text())
answer = grader.loads(answer_path.read_text()) if answer_path.is_file() and not answer_path.is_symlink() else {}
result = grader.grade(expected["poles"], expected["fs_required"], answer)
Path("/logs/verifier/metrics.json").write_text(json.dumps(result))
Path("/logs/verifier/reward.txt").write_text(str(result["reward"]))
"""

SOLVE_SH = """#!/bin/bash
set -e
mkdir -p /app/output
cd /solution
python3 -c "
import json
from geotech_rl.solver import solve
json.dump(solve('/app/data'), open('/app/output/pole_check.json', 'w'), indent=1)"
"""


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")
    path.chmod(0o755)


def export(seed: int, level: int, root: Path) -> Instance:
    package = Path(__file__).parent
    name = f"poles-n{level}-s{seed:05d}"
    task = root / name
    (task / "tests").mkdir(parents=True, exist_ok=True)
    (task / "solution").mkdir(exist_ok=True)
    data = task / "environment" / "data"
    built = build(seed, level, data)
    (task / "environment" / "Dockerfile").write_text(ENV_DOCKERFILE, encoding="utf-8")
    (task / "task.toml").write_text(TASK_TOML.format(name=name, level=level), encoding="utf-8")
    (task / "instruction.md").write_text(records.instruction(built.scenario), encoding="utf-8")
    (task / "tests" / "Dockerfile").write_text(TEST_DOCKERFILE, encoding="utf-8")
    _write_executable(task / "tests" / "test.sh", TEST_SH)
    (task / "tests" / "verify.py").write_text(VERIFY_PY, encoding="utf-8")
    shutil.copy(package / "grader.py", task / "tests" / "grader.py")
    truth = dict(poles=built.truth, fs_required=built.scenario.fs_required)
    (task / "tests" / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    _write_executable(task / "solution" / "solve.sh", SOLVE_SH)
    shutil.copytree(package, task / "solution" / "geotech_rl", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "harbor.py", "cli.py"))
    return built
