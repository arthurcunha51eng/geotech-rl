"""Writes oracle demonstrations for a supervised warm start: the episode a careful engineer would run.

Rollouts with malformed tool calls carry no reward signal, so GRPO cannot start from them. Every tool result here
comes from the real tools, so the model trains on the text it will meet in an episode, and every demonstration is
graded before it is kept. Instances come from the warm-up levels of the train split only. The script holds no decision
on purpose: it reads no rain, no unit and no remark, so whatever the model gains above the warm-up comes from the reward.

    python demonstrations.py <output dir> <episodes per level> [levels, default 0,1]
"""
import asyncio
import json
import re
import sys
from pathlib import Path

from geotech_rl import grader, quality, sandbox
from geotech_poles import taskset
from geotech_poles.servers import poles

CODE = """import pandas as pd

control = pd.read_csv('compaction_control.csv')
gamma = (9.81 * control['dry_density_g_cm3'] * (1 + control['water_content_pct'] / 100)).mean()

tests = pd.read_csv('pull_tests.csv')
constants = []
for _, test in tests.iterrows():
    sheet = pd.read_csv('pull_tests/' + test['test'] + '.csv')
    H = sheet.loc[sheet['held'] == 'yes', 'load_daN'].max() / 100
    L, D, e = test['embedded_m'], test['collar_diam_m'], test['cable_height_m']
    print(test['test'], 'H', H)
    constants.append(H / (0.5 * gamma * D * L**3 / (e + L)))
K = sum(constants) / len(constants)
print('gamma', round(gamma, 3), 'K', round(K, 3))

poles = pd.read_csv('as_built_poles.csv')
for _, pole in poles.iterrows():
    L, D = pole['embedded_m'], pole['collar_diam_m']
    e = {length} - L - 0.10
    H = K * 0.5 * gamma * D * L**3 / (e + L)
    fs = H * 100 / pole['nominal_daN']
    print(pole['pole'], round(fs, 3), bool(fs < {required}))
"""

WORKERS = 32
TOOLS = [
    dict(type="function", function=dict(
        name="site_read_file", description=poles.PolesToolset.read_file.__doc__,
        parameters=dict(type="object", properties=dict(path=dict(title="Path", type="string")), required=["path"],
                        title="read_fileArguments"))),
    dict(type="function", function=dict(
        name="site_run_python", description=poles.PolesToolset.run_python.__doc__,
        parameters=dict(type="object", properties=dict(code=dict(title="Code", type="string")), required=["code"],
                        title="run_pythonArguments"))),
    dict(type="function", function=dict(
        name="site_submit", description=poles.PolesToolset.submit.__doc__,
        parameters=dict(type="object", properties=dict(answer=dict(additionalProperties=True, title="Answer", type="object")),
                        required=["answer"], title="submitArguments"))),
]


class Episode:
    """Replays the tools of one episode against the task's records."""

    def __init__(self, task: taskset.PolesTask):
        self.task = task
        self.state = poles.PolesState(files=task.data.files)
        self.messages = [dict(role="user", content=task.data.prompt)]
        self.count = 0

    def call(self, name: str, arguments: dict) -> dict:
        self.count += 1
        return dict(id=f"call_{self.count}", type="function",
                    function=dict(name=name, arguments=json.dumps(arguments)))

    def reply(self, calls: list[dict], results: list[str]) -> None:
        self.messages.append(dict(role="assistant", content="", tool_calls=calls))
        self.messages += [dict(role="tool", tool_call_id=c["id"], content=r) for c, r in zip(calls, results)]

    def read(self, *paths: str) -> None:
        results = []
        for path in paths:
            text = self.task.data.files[path]
            results.append(sandbox.clip(text, poles.READ_LIMIT) + poles.budget_note(self.state))
        self.reply([self.call("site_read_file", dict(path=p)) for p in paths], results)

    async def run(self, code: str) -> str:
        output = await sandbox.run(code, poles.materialise(self.state))
        self.reply([self.call("site_run_python", dict(code=code))], [output + poles.budget_note(self.state)])
        return output

    def submit(self, answer: dict) -> None:
        self.messages.append(dict(role="assistant", content="", tool_calls=[self.call("site_submit", dict(answer=answer))]))


def answer_from(output: str) -> dict:
    rows = re.findall(r"^(L-\d+) ([\d.]+) (True|False)$", output, re.M)
    return {pole: dict(fs=round(float(fs), 2), refound=refound == "True") for pole, fs, refound in rows}


async def demonstrate(task: taskset.PolesTask) -> list[dict] | None:
    readme = task.data.files["README.md"]
    length = re.search(r"All poles are (\d+(?:\.\d+)?) m", readme).group(1)
    required = re.search(r"Required safety factor: (\d+(?:\.\d+)?)", readme).group(1)
    sheet = sorted(name for name in task.data.files if name.startswith("pull_tests/"))[0]
    episode = Episode(task)
    try:
        episode.read("README.md", "as_built_poles.csv", "compaction_control.csv")
        episode.read("pull_tests.csv", sheet, "rain_daily.csv")
        output = await episode.run(CODE.replace("{length}", length).replace("{required}", required))
        answer = answer_from(output)
        episode.submit(answer)
    finally:
        poles.release(episode.state)
    return episode.messages if grader.grade(task.data.truth, task.data.fs_required, answer)["reward"] == 1.0 else None


async def main(out: Path, per_level: int, levels: list[int]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    limit = asyncio.Semaphore(WORKERS)

    async def one(row: dict) -> list[dict] | None:
        async with limit:
            return await demonstrate(taskset._task(row, taskset.PolesTaskConfig()))

    rows = [r for level in levels for r in quality.read_manifest(taskset.MANIFESTS / f"train_n{level}.jsonl")[:per_level]]
    results = await asyncio.gather(*(one(row) for row in rows))
    kept = [messages for messages in results if messages is not None]
    rejected = len(results) - len(kept)
    with (out / "train.jsonl").open("w", encoding="utf-8") as f:
        for messages in kept:
            f.write(json.dumps(dict(messages=messages, tools=json.dumps(TOOLS))) + "\n")
    kept = len(kept)
    print(f"DEMONSTRATIONS kept {kept} rejected {rejected}")
    if kept == 0 or rejected > 0.02 * kept:
        sys.exit("DEMONSTRATIONS_FAILED: the scripted oracle does not solve its own instances")


if __name__ == "__main__":
    levels = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 1]
    asyncio.run(main(Path(sys.argv[1]), int(sys.argv[2]), levels))
