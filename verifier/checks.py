"""Checks shared by the tool verifiers. Pure functions of their arguments and the files on disk."""

import json
import re
from pathlib import Path

from .rule import DECIMAL_RULE, parse_decimal, qualifies

LEDGER = "score_ledger.json"
CANDIDATES = "candidates"
HYPERPLANES = 256  # how many hyperplanes src/plugin/data/hyperplanes.json holds


def result(valid, message, *, score=None, **details):
    return {
        "valid": valid,
        "score": float(valid) if score is None else score,
        "message": message,
        "details": details,
    }


def decimal_problem(text, name):
    if parse_decimal(text) is None:
        return f"{name} must be {DECIMAL_RULE}, got {text!r}"
    return None


def run_id_problem(dataset, run_id):
    if not re.fullmatch(re.escape(dataset.value) + r"-[1-9][0-9]*x[1-9][0-9]*", run_id):
        return f"run_id {run_id!r} is not a {dataset.value} run, {dataset.value}-<bits_per_band>x<bands>"
    return None


def read_json(path):
    """The JSON in ``path``, or None when there is no such file."""
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def scored(pair):
    """A contract ScoredPair as the ledger records it."""
    return {"a": pair.a, "b": pair.b, "dot": pair.dot, "norm_a": pair.norm_a, "norm_b": pair.norm_b}


def qualifying_problem(pair, threshold):
    """What is wrong with a pair, as the ledger records it, said to qualify at ``threshold`` (a Fraction)."""
    name = f"({pair['a']!r}, {pair['b']!r})"
    if not pair["a"] < pair["b"]:
        return f"pair {name} is not in canonical order, a before b"
    if pair["norm_a"] <= 0 or pair["norm_b"] <= 0:
        return f"pair {name} has a squared length that is not positive"
    if not qualifies(pair["dot"], pair["norm_a"], pair["norm_b"], threshold):
        return f"pair {name} does not meet the threshold by its own dot and norms"
    return None
