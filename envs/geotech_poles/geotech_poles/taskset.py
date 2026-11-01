"""Inspect pole foundations from site records, with file and Python tools and a short turn budget.

Manifests fix the instances: every split is a filtered list of seeds, and train and held-out never share one.
Truth lives in TaskData on the environment side and is never written to the rollout's directory. Submission
returns no score, so an episode gives the agent no feedback to probe the grader with.
"""
import os
import tempfile
from pathlib import Path

import geotech_rl
from geotech_rl import grader, instance, quality, records, solver
from geotech_poles.servers.poles import TOOL_BUDGET, PolesState, PolesToolset, release

import verifiers.v1 as vf

MAX_TURNS = TOOL_BUDGET     # every turn spends a call, so the budget shown to the agent is the limit that binds
MANIFESTS = Path(os.environ.get("GEOTECH_MANIFESTS", Path(geotech_rl.__file__).parents[2] / "manifests"))
SYSTEM = (
    "You are a geotechnical engineer inspecting a site. Read the records you need, compute with site_run_python, "
    f"and finish by calling site_submit. You have {TOOL_BUDGET} tool calls; an episode without site_submit scores zero.\n\n"
)


class PolesData(vf.TaskData):
    seed: int
    level: int
    control: str
    fs_required: float
    truth: dict
    files: dict[str, str]


class PolesTaskConfig(vf.TaskConfig):
    reward: str = "binary"
    """binary trains on the full check; partial trains on the share of poles within tolerance."""


class PolesTask(vf.Task[PolesData, PolesState, PolesTaskConfig]):
    async def setup(self, trace: vf.Trace) -> None:
        trace.state.files = self.data.files

    @vf.stop
    async def done(self, trace: vf.Trace) -> bool:
        return bool(trace.state.answer) or trace.num_turns >= MAX_TURNS or trace.state.calls >= TOOL_BUDGET

    async def finalize(self, trace: vf.Trace) -> None:
        release(trace.state)

    def _graded(self, trace: vf.Trace) -> dict:
        return grader.grade(self.data.truth, self.data.fs_required, grader.loads(trace.state.answer))

    @vf.reward(weight=1.0)
    async def inspection(self, trace: vf.Trace) -> float:
        return self._graded(trace)[self.config.reward if self.config.reward == "partial" else "reward"]

    @vf.metric
    async def partial(self, trace: vf.Trace) -> float:
        return self._graded(trace)["partial"]

    @vf.metric
    async def decision_ok(self, trace: vf.Trace) -> float:
        return self._graded(trace)["decision_ok"]

    @vf.metric
    async def submitted(self, trace: vf.Trace) -> float:
        return float(bool(trace.state.answer))

    async def validate(self) -> bool:
        with tempfile.TemporaryDirectory() as tmp:
            for name, text in self.data.files.items():
                path = Path(tmp) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            answer = solver.solve(Path(tmp))
        return grader.grade(self.data.truth, self.data.fs_required, answer)["reward"] == 1.0


class PolesConfig(vf.TasksetConfig):
    levels: list[int] = [1, 2, 3]
    split: str = "train"
    num_seeds: int = 1000
    inline: bool = False
    """Puts every record in the prompt, so the episode needs no reading turns."""
    tools: vf.SharedToolsetConfig = vf.SharedToolsetConfig()
    task: PolesTaskConfig = PolesTaskConfig()


def _task(row: dict, config: PolesTaskConfig, inline: bool = False) -> PolesTask:
    seed, level = row["seed"], row["level"]
    with tempfile.TemporaryDirectory() as tmp:
        built = instance.build(seed, level, Path(tmp), row["split"], attempt=row["attempt"])
        files = {p.relative_to(tmp).as_posix(): p.read_text(encoding="utf-8")
                 for p in sorted(Path(tmp).rglob("*")) if p.is_file()}
    s = built.scenario
    listing = "Files: " + ", ".join(files) + "\n"
    if inline:
        listing = "The records are below and are also in the working directory.\n\n" + "\n".join(
            f"--- {name} ---\n{text}" for name, text in files.items())
    return PolesTask(
        PolesData(prompt=SYSTEM + records.instruction(s, tools=True) + listing, seed=seed, level=level,
                  control=s.control, fs_required=s.fs_required, truth=built.truth, files=files),
        config)


class PolesTaskset(vf.Taskset[PolesTask, PolesConfig]):
    @classmethod
    def toolsets(cls, config: PolesConfig) -> list[vf.Toolset]:
        return [PolesToolset(config.tools)]

    def load(self) -> list[PolesTask]:
        rows = []
        for level in self.config.levels:
            rows += quality.read_manifest(MANIFESTS / f"{self.config.split}_n{level}.jsonl")[:self.config.num_seeds]
        return [_task(row, self.config.task, self.config.inline) for row in rows]
