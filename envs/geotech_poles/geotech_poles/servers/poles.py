"""Tools of one inspection episode. The server is shared by a worker; each rollout keeps its files in self.state."""
import json
import shutil
import tempfile
from pathlib import Path

from geotech_rl import sandbox

import verifiers.v1 as vf

READ_LIMIT = 6000
TOOL_BUDGET = 16           # calls per episode, shown to the agent so it can plan the submission
URGENT = 3


class PolesState(vf.State):
    files: dict[str, str] = {}
    workdir: str = ""
    answer: str = ""
    calls: int = 0


def materialise(state: PolesState) -> Path:
    """Writes the rollout's records on first use, so concurrent rollouts never share a directory."""
    if state.workdir:
        return Path(state.workdir)
    workdir = Path(tempfile.mkdtemp(prefix="poles-"))
    for name, text in state.files.items():
        path = workdir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    state.workdir = str(workdir)
    return workdir


def budget_note(state: PolesState) -> str:
    """Counts one call and tells the agent what is left; without it, weak models never reach the submission."""
    state.calls += 1
    left = TOOL_BUDGET - state.calls
    note = f"\n[{left} tool calls left]"
    return note + " Call site_submit now with your best answer." if left <= URGENT else note


def release(state: PolesState) -> None:
    if state.workdir:
        shutil.rmtree(state.workdir, ignore_errors=True)


class PolesToolset(vf.Toolset[vf.SharedToolsetConfig, PolesState]):
    TOOL_PREFIX = "site"

    @vf.tool
    async def read_file(self, path: str) -> str:
        """Read one site record. Paths are relative to the working directory."""
        root = materialise(self.state).resolve()
        target = (root / path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            return f"no such file: {path}" + budget_note(self.state)
        return sandbox.clip(target.read_text(encoding="utf-8"), READ_LIMIT) + budget_note(self.state)

    @vf.tool
    async def run_python(self, code: str) -> str:
        """Run Python 3 in the working directory, with numpy, pandas and scipy. Print what you need to see."""
        return await sandbox.run(code, materialise(self.state)) + budget_note(self.state)

    @vf.tool
    async def submit(self, answer: dict) -> str:
        """Submit the final answer as an object keyed by pole ID. This ends the inspection."""
        self.state.answer = json.dumps(answer)
        return "recorded"


if __name__ == "__main__":
    PolesToolset.run()
