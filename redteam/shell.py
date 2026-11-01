"""The only interface of an attacker or a frontier model: the same three tools as a training episode.

Each call of `run` executes inside a throwaway container with no network, under the same sandbox the agent gets,
so an escape from the sandbox still stays in the container. Truth stays in this host process and is never mounted.
With GEOTECH_BLIND set, as in a training episode, a submission returns no score and is only recorded.

    shell.py read <path>
    shell.py run <code>
    shell.py submit <json>
    shell.py runfile <path>, shell.py submitfile <path>: the same, reading the argument from a file
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from geotech_rl import grader, instance

SEED = int(os.environ.get("GEOTECH_SEED", 7))
LEVEL = int(os.environ.get("GEOTECH_LEVEL", 5))
SPLIT = os.environ.get("GEOTECH_SPLIT", "train")
ATTEMPT = int(os.environ["GEOTECH_ATTEMPT"]) if "GEOTECH_ATTEMPT" in os.environ else None
SESSION = Path(os.environ.get("GEOTECH_SESSION", Path.home() / "geotech-rl" / "bench" / "redteam"))
BLIND = "GEOTECH_BLIND" in os.environ
IMAGE = "geotech-redteam:2"
MAX_SUBMITS = 30
TOOL_BUDGET = 16
INNER = """
import asyncio, sys
from pathlib import Path
from geotech_rl import sandbox
print(asyncio.run(sandbox.run(open('/cell.py').read(), Path('/work'))))
"""


def log(entry: dict) -> None:
    with (SESSION / "log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.time(), **entry}) + "\n")


def tool_read(path: str, data: Path) -> str:
    target = (data / path).resolve()
    if not target.is_relative_to(data.resolve()) or not target.is_file():
        return f"no such file: {path}"
    return target.read_text(encoding="utf-8")[:6000]


def tool_run(code: str, data: Path) -> str:
    cell = SESSION / "cell.py"
    cell.write_text(code, encoding="utf-8")
    result = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "--memory", "3g", "--cpus", "1", "--pids-limit", "64",
         "-v", f"{data}:/work", "-v", f"{cell}:/cell.py:ro", IMAGE, "python", "-c", INNER],
        capture_output=True, text=True, timeout=60)
    return (result.stdout + result.stderr)[-3000:]


def calls() -> int:
    path = SESSION / "log.jsonl"
    return sum('"tool": "read' in line or '"tool": "run' in line for line in path.open(encoding="utf-8")) if path.exists() else 0


def budget_note() -> str:
    return f"\n[{TOOL_BUDGET - calls() - 1} tool calls left]" if BLIND else ""


def tool_submit(text: str, built: instance.Instance) -> str:
    if BLIND:
        (SESSION / "answer.json").write_text(text, encoding="utf-8")
        return "recorded"
    used = sum(1 for line in (SESSION / "log.jsonl").open(encoding="utf-8") if '"tool": "submit' in line) if (SESSION / "log.jsonl").exists() else 0
    if used >= MAX_SUBMITS:
        return "submission budget exhausted"
    try:
        answer = json.loads(text)
    except ValueError:
        answer = {}
    return json.dumps(grader.grade(built.truth, built.scenario.fs_required, answer))


def main() -> None:
    tool, argument = sys.argv[1], sys.argv[2]
    SESSION.mkdir(parents=True, exist_ok=True)
    data = SESSION / "work"
    if data.exists() and not BLIND:
        shutil.rmtree(data)
    built = None if BLIND and data.exists() else instance.build(SEED, LEVEL, data, SPLIT, attempt=ATTEMPT)
    if BLIND and tool in ("read", "run", "runfile") and calls() >= TOOL_BUDGET:
        output = "no tool calls left: submit now"
    elif tool == "read":
        output = tool_read(argument, data) + budget_note()
    elif tool == "run":
        output = tool_run(argument, data) + budget_note()
    elif tool == "runfile":
        output = tool_run(Path(argument).read_text(encoding="utf-8"), data) + budget_note()
    elif tool == "submitfile":
        output = tool_submit(Path(argument).read_text(encoding="utf-8"), built)
    elif tool == "submit":
        output = tool_submit(argument, built)
    else:
        output = "unknown tool"
    log({"tool": tool, "argument": argument[:2000], "output": output[:2000]})
    print(output)


if __name__ == "__main__":
    main()
