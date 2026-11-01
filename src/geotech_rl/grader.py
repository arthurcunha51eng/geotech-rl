"""Grades an answer against the truth. The binary reward trains; partial and decision_ok are diagnostics."""
import json
import math

REL_TOLERANCE = 0.10
FLAG_FREE = 0.10            # a pole this close to the requirement may be flagged either way
FS_KEYS = ("fs", "FS", "safety_factor")
FLAG_KEYS = ("refound", "re_found", "refound_required", "refounding_required")


def _no_duplicates(pairs: list) -> dict:
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate key")
    return dict(pairs)


def loads(text: str):
    """The answer as submitted; anything that is not one clean JSON value counts as no answer."""
    try:
        return json.loads(text, object_pairs_hook=_no_duplicates)
    except (ValueError, RecursionError):
        return {}


def normalise_id(pid) -> str:
    s = str(pid).strip().upper().replace("_", "-").replace(" ", "")
    if s.startswith("L") and not s.startswith("L-"):
        s = "L-" + s[1:]
    head, _, num = s.partition("-")
    return f"{head}-{int(num):02d}" if num.isdigit() else s


def _one(entry: dict, keys: tuple):
    found = [k for k in keys if k in entry]
    return entry[found[0]] if len(found) == 1 else None


def parse(answer) -> dict:
    """Keeps only well-formed entries; a malformed pole simply counts as missing."""
    if not isinstance(answer, dict):
        return {}
    parsed = {}
    for key, entry in answer.items():
        pid = normalise_id(key)
        if pid in parsed or not isinstance(entry, dict):
            continue
        fs, flag = _one(entry, FS_KEYS), _one(entry, FLAG_KEYS)
        number = isinstance(fs, (int, float)) and not isinstance(fs, bool) and math.isfinite(fs)
        if number and isinstance(flag, bool):
            parsed[pid] = (float(fs), flag)
    return parsed


def worst_error(truth: dict, answer) -> float:
    """Largest relative error of a safety factor; the margin of a route, in the units of the tolerance."""
    parsed = parse(answer)
    if any(pid not in parsed for pid in truth):
        return math.inf
    return max(abs(parsed[pid][0] / entry["fs"] - 1.0) for pid, entry in truth.items())


def grade(truth: dict, required: float, answer) -> dict:
    parsed = parse(answer)
    fs_ok, flag_ok = 0, 0
    for pid, entry in truth.items():
        if pid not in parsed:
            continue
        fs, flag = parsed[pid]
        fs_ok += abs(fs - entry["fs"]) <= REL_TOLERANCE * entry["fs"]
        flag_free = abs(entry["fs"] - required) <= FLAG_FREE * required
        flag_ok += flag_free or flag == entry["refound"]
    n = len(truth)
    return dict(reward=float(fs_ok == n and flag_ok == n), partial=fs_ok / n, decision_ok=float(flag_ok == n))
