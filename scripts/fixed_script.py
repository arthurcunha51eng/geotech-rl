"""Runs the demonstration script, unchanged, on held-out instances: the baseline any score has to beat.

A level this script solves holds no decision, whatever a model scores there.

    python fixed_script.py <output.json> [instances per level, default 60]
"""
import asyncio
import json
import re
import sys
import tempfile
from pathlib import Path

from demonstrations import CODE, answer_from

from geotech_rl import grader, instance, quality, sandbox, scenario

MANIFESTS = Path(__file__).parents[1] / "manifests"
WORKERS = 16


async def attempt(row: dict, limit: asyncio.Semaphore) -> tuple[bool, str]:
    async with limit:
        with tempfile.TemporaryDirectory() as tmp:
            built = instance.build(row["seed"], row["level"], Path(tmp), row["split"], attempt=row["attempt"])
            readme = (Path(tmp) / "README.md").read_text(encoding="utf-8")
            length = re.search(r"All poles are (\d+(?:\.\d+)?) m", readme).group(1)
            required = re.search(r"Required safety factor: (\d+(?:\.\d+)?)", readme).group(1)
            output = await sandbox.run(CODE.replace("{length}", length).replace("{required}", required), Path(tmp))
    result = grader.grade(built.truth, built.scenario.fs_required, answer_from(output))
    return result["reward"] == 1.0, row["control"]


async def main(out: Path, count: int) -> None:
    limit = asyncio.Semaphore(WORKERS)
    report = {}
    for level in scenario.LEVELS:
        rows = quality.read_manifest(MANIFESTS / f"heldout_n{level}.jsonl")[:count]
        results = await asyncio.gather(*(attempt(row, limit) for row in rows))
        plain = [solved for solved, control in results if not control]
        report[f"n{level}"] = dict(solved=sum(solved for solved, _ in results), episodes=len(results),
                                   solved_outside_controls=sum(plain), episodes_outside_controls=len(plain))
        print(f"N{level}", report[f"n{level}"], flush=True)
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 60))
