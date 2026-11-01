"""Measures a frontier model on held-out instances through the training tools, with no score returned to it.

Its outputs only measure: they never become training data. Each instance runs in its own session directory, and the
answer is graded here, on the host, where the truth is.

Needs Docker, the image from redteam/Dockerfile tagged geotech-redteam:2, and an executable at ~/geotech-rl/rt that
runs redteam/shell.py with its arguments.

    python frontier.py --levels 5,6,7,8 --count 4 --model claude-opus-5-5 --effort medium --out results.jsonl
"""
import argparse
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from geotech_rl import grader, instance, quality, records

ROOT = Path(__file__).parents[1]
TOOL = Path.home() / "geotech-rl" / "rt"
SESSIONS = Path.home() / "geotech-rl" / "bench" / "frontier"
PROMPT = """You are a geotechnical engineer inspecting a site. Read the records you need, compute with Python, and finish \
by submitting. You have 16 tool calls; an episode without a submission scores zero.

{task}{files}

Your only tools are these commands:
  {tool} read <path>        reads a record, relative to the working directory
  {tool} run '<python>'     runs Python 3 in the working directory, with numpy, pandas and scipy
  {tool} submit '<json>'    submits the final answer as a JSON object keyed by pole ID. This ends the inspection.
"""


def episode(job: dict) -> dict:
    row, args = job["row"], job["args"]
    session = SESSIONS / f"{args.tag}-n{row['level']}-{row['seed']}"
    session.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GEOTECH_SEED=str(row["seed"]), GEOTECH_LEVEL=str(row["level"]), GEOTECH_SPLIT=row["split"],
               GEOTECH_ATTEMPT=str(row["attempt"]), GEOTECH_SESSION=str(session), GEOTECH_BLIND="1")
    with tempfile.TemporaryDirectory() as tmp:
        built = instance.build(row["seed"], row["level"], Path(tmp), row["split"], attempt=row["attempt"])
        files = "Files: " + ", ".join(sorted(p.relative_to(tmp).as_posix() for p in Path(tmp).rglob("*") if p.is_file())) + "\n"
        prompt = PROMPT.format(task=records.instruction(built.scenario, tools=True), files=files, tool=TOOL)
        started = time.time()
        try:
            run = subprocess.run(
                ["npx", "-y", "@anthropic-ai/claude-code", "-p", prompt, "--model", args.model, "--effort", args.effort,
                 "--max-turns", "40", "--output-format", "json", "--allowedTools", f"Bash({TOOL}:*)",
                 "--disallowedTools", "Read,Write,Edit,Glob,Grep,WebFetch,WebSearch,Agent,Task,TodoWrite"],
                cwd=session, env=env, capture_output=True, text=True, timeout=args.timeout, stdin=subprocess.DEVNULL)
            output, code = run.stdout, run.returncode
        except subprocess.TimeoutExpired:
            output, code = "", -1
        path = session / "answer.json"
        answer = grader.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        result = grader.grade(built.truth, built.scenario.fs_required, answer)
        route = instance.diagnose(Path(tmp), answer)
    usage = {}
    try:
        report = json.loads(output)
        usage = dict(cost_usd=report.get("total_cost_usd"), turns=report.get("num_turns"), usage=report.get("usage"))
    except ValueError:
        pass
    return dict(level=row["level"], seed=row["seed"], control=row["control"], reward=result["reward"], partial=result["partial"],
                decision_ok=result["decision_ok"], route=route, submitted=path.exists(), exit=code,
                seconds=round(time.time() - started), model=args.model, effort=args.effort, **usage)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", default="5,6,7,8")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--tag", default="ref")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = [r for level in args.levels.split(",")
            for r in quality.read_manifest(ROOT / "manifests" / f"heldout_n{level}.jsonl")[args.skip:args.skip + args.count]]
    jobs = [dict(row=r, args=args) for r in rows]
    with ThreadPoolExecutor(args.workers) as pool, open(args.out, "a", encoding="utf-8") as out:
        for result in pool.map(episode, jobs):
            out.write(json.dumps(result) + "\n")
            out.flush()
            print({k: result[k] for k in ("level", "seed", "control", "reward", "partial", "route", "submitted", "seconds")}, flush=True)


if __name__ == "__main__":
    main()
