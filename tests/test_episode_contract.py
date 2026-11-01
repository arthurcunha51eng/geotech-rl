import asyncio
import inspect

import pytest

poles = pytest.importorskip("geotech_poles.servers.poles")


def test_submit_returns_no_score():
    """Repeated submissions with feedback would let a policy search for the answer; a real episode has neither."""
    source = inspect.getsource(poles.PolesToolset.submit)
    assert "grade" not in source and 'return "recorded"' in source


def test_submission_is_an_object_not_json_inside_json():
    """A string argument forced quotes to be escaped; 23 of 67 submit calls failed on that alone."""
    assert "answer: dict" in inspect.getsource(poles.PolesToolset.submit)


def test_episode_stops_at_first_submission():
    taskset = pytest.importorskip("geotech_poles.taskset")
    source = inspect.getsource(taskset.PolesTask.done)
    assert "trace.state.answer" in source and "MAX_TURNS" in source


def test_the_budget_told_to_the_agent_is_the_limit_that_ends_the_episode():
    """Cutting at 12 turns while the note said 7 calls were left ended 17 of 60 episodes without warning."""
    taskset = pytest.importorskip("geotech_poles.taskset")
    assert taskset.MAX_TURNS >= poles.TOOL_BUDGET


def test_budget_note_counts_down_and_urges_submission():
    state = poles.PolesState()
    first = poles.budget_note(state)
    assert f"[{poles.TOOL_BUDGET - 1} tool calls left]" in first and "site_submit" not in first
    state.calls = poles.TOOL_BUDGET - poles.URGENT - 1
    assert "site_submit" in poles.budget_note(state)


def test_rollouts_do_not_share_a_directory():
    first, second = poles.PolesState(files={"a.txt": "1"}), poles.PolesState(files={"a.txt": "2"})
    a, b = poles.materialise(first), poles.materialise(second)
    try:
        assert a != b and (a / "a.txt").read_text() == "1" and (b / "a.txt").read_text() == "2"
    finally:
        poles.release(first)
        poles.release(second)
