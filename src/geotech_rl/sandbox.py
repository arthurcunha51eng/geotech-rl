"""Runs agent code in a working directory: minimal environment, resource limits and an audit hook."""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

WALL_SECONDS = 20.0
OUTPUT_LIMIT = 1500
SILENT = "(no output: nothing was printed. Use print() to see results.)"
BOOT = Path(__file__).with_name("sandbox_boot.py")
ENVIRONMENT = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "HOME": "/nonexistent"}


def clip(text: str, limit: int) -> str:
    """Keeps both ends: the end of a traceback is the part the agent needs."""
    if len(text) <= limit:
        return text
    tail = limit // 3
    return text[:limit - tail] + f"\n... [{len(text) - limit} characters cut]\n" + text[-tail:]


async def run(code: str, root: Path, wall_seconds: float = WALL_SECONDS) -> str:
    """Output of the cell, or the error it raised, for the agent to read."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as cell:
        cell.write(code)
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-I", str(BOOT), cell.name, cwd=root, env=ENVIRONMENT,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
    try:
        output, _ = await asyncio.wait_for(process.communicate(), wall_seconds)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return f"stopped after {wall_seconds:g} s"
    finally:
        os.unlink(cell.name)
    return clip(output.decode(errors="replace"), OUTPUT_LIMIT) or SILENT
