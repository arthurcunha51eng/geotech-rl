"""Reads the traces of an evaluation and summarises them per level.

    python summarise_eval.py <output dir> [--strict]

--strict fails when the environment, not the model, broke an episode: the run stops before more money goes.
"""
import collections
import glob
import json
import math
import re
import sys

import zstandard


HARNESS_TOLERANCE = 0.05    # a rare server error is noise; more than this is the bench failing
UNPARSED_TOLERANCE = 0.10   # above this, sampling or the tool format is ending episodes, not the task


def computed(answer, cells: list[str]) -> bool:
    """Safety factors that never appeared in the output of a cell were invented, not calculated."""
    printed = [float(x) for cell in cells for x in re.findall(r"\d+\.\d+", cell)]
    values = [float(x) for x in re.findall(r'fs\\*"\s*:\s*(\d+\.?\d*)', str(answer))]
    return bool(values) and all(any(abs(p - v) <= 0.01 * v for p in printed) for v in values)


def tool_failures(trace: dict) -> dict:
    nodes = trace.get("nodes", [])
    nodes = nodes if isinstance(nodes, list) else list(nodes.values())
    messages = [n.get("message", {}) for n in nodes]
    calls = {c["id"]: c for m in messages for c in m.get("tool_calls") or []}
    replies = [(calls.get(m.get("tool_call_id"), {}).get("name"), m.get("content")) for m in messages if m.get("role") == "tool"]
    outputs = [text for _, text in replies if isinstance(text, str)]
    cells = [text for name, text in replies if name == "site_run_python" and isinstance(text, str)]
    answers = [c["arguments"] for c in calls.values() if c["name"] == "site_submit"]
    return dict(tool_results=len(outputs), sandbox_blocks=sum("blocked in this sandbox" in o for o in outputs),
                python_errors=sum("Traceback" in o for o in outputs), silent_cells=sum(o.startswith("(no output") for o in cells),
                computed=bool(answers) and bool(trace.get("metrics", {}).get("submitted")) and computed(answers[-1], cells))


def ending(trace: dict) -> str:
    """Why an episode ended. Anything but a submission or the agent's own stop is a limit imposed on it."""
    calls = trace.get("calls", [])
    last = calls[-1] if calls else {}
    if "maximum context length" in str(last.get("error")):
        return "context_overflow"
    if trace.get("stop_condition") == "harness_error":
        return "harness_error"
    if trace.get("metrics", {}).get("submitted"):
        return "submitted"
    if trace.get("stop_condition") == "done":
        return "budget_spent"
    nodes = trace.get("nodes", [])
    nodes = nodes if isinstance(nodes, list) else list(nodes.values())
    if nodes and not nodes[-1].get("message", {}).get("content"):
        return "unparsed_reply"     # the parser drops malformed tool calls and leaves an empty message
    return "stopped_talking"


def wilson(successes: float, n: int, z: float = 1.96) -> list[float]:
    """95% interval of a proportion; honest at zero successes, where a normal interval collapses."""
    if n == 0:
        return [0.0, 0.0]
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def episodes(root: str):
    for path in glob.glob(f"{root}/**/traces/stream/*", recursive=True):
        with open(path, "rb") as f:
            text = zstandard.ZstdDecompressor().stream_reader(f).read().decode() if path.endswith(".zst") else f.read().decode()
        for line in text.splitlines():
            if line.strip():
                yield json.loads(line)


def summarise(root: str) -> dict:
    rows = collections.defaultdict(list)
    for episode in episodes(root):
        for trace in episode["traces"]:
            data = trace["task"]["data"]
            metrics = trace.get("metrics", {})
            rows[data["level"]].append(dict(
                reward=sum(r["score"] * r["weight"] for r in trace.get("rewards", {}).values()),
                partial=metrics.get("partial", 0.0), decision_ok=metrics.get("decision_ok", 0.0),
                submitted=metrics.get("submitted", 0.0), control=bool(data.get("control")),
                stop=str(trace.get("stop_condition")), ok=bool(trace.get("ok")),
                calls=len(trace.get("calls", [])), tokens_in=trace.get("num_input_tokens", 0),
                tokens_out=trace.get("num_output_tokens", 0), ending=ending(trace),
                truncated=sum(c.get("finish_reason") == "length" for c in trace.get("calls", [])),
                model_seconds=trace.get("timing", {}).get("agent", {}).get("model", {}).get("duration", 0.0),
                **tool_failures(trace)))
    report = {}
    for level, items in sorted(rows.items()):
        n = len(items)
        solved = sum(i["reward"] >= 1.0 for i in items)
        mean = lambda key: round(sum(i[key] for i in items) / n, 3)
        report[level] = dict(
            episodes=n, reward=round(solved / n, 3), reward_ci95=wilson(solved, n),
            partial=mean("partial"), decision_ok=mean("decision_ok"), submitted=mean("submitted"),
            model_calls=mean("calls"), tokens_in=mean("tokens_in"), tokens_out=mean("tokens_out"),
            env_errors=sum(not i["ok"] for i in items),
            sandbox_blocks=sum(i["sandbox_blocks"] for i in items), python_errors=sum(i["python_errors"] for i in items),
            tool_results=sum(i["tool_results"] for i in items), silent_cells=sum(i["silent_cells"] for i in items),
            computed_answers=sum(i["computed"] for i in items),
            truncated_replies=sum(i["truncated"] for i in items), model_seconds=mean("model_seconds"),
            endings=dict(collections.Counter(i["ending"] for i in items).most_common()),
            stops=dict(collections.Counter(i["stop"] for i in items).most_common(4)))
    return report


if __name__ == "__main__":
    report = summarise(sys.argv[1])
    print(json.dumps(report, indent=1))
    if "--strict" in sys.argv:
        broken = {k: v["endings"] | dict(sandbox_blocks=v["sandbox_blocks"]) for k, v in report.items()
                  if v["sandbox_blocks"] or v["endings"].get("context_overflow")
                  or v["endings"].get("harness_error", 0) > HARNESS_TOLERANCE * v["episodes"]
                  or v["endings"].get("unparsed_reply", 0) > UNPARSED_TOLERANCE * v["episodes"]}
        if broken or not report:
            sys.exit(f"ENVIRONMENT_FAILED {broken or 'no traces'}")
