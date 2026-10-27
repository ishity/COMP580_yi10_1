"""score_pairs: score a rank range of an LSH run with the exact rule, within the task's budget.

Every call is recorded in score_ledger.json in the workspace. The first call fixes the ledger's dataset,
threshold and budget; later calls must pass the same. A pair any earlier call scored is skipped, neither
scored nor charged again, so the budget counts distinct pairs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from contract import ScoredPair, ScoreInput, ScoreOutput
from plugin.bundle import LEDGER, candidates_path, decimal, dot, load
from plugin.rule import parse_decimal, qualifies


def _ledger(given: ScoreInput, workspace: Path) -> dict:
    """The ledger so far, or a new one fixing this call's dataset, threshold and budget."""
    path = workspace / LEDGER
    if not path.exists():
        return {"dataset": given.dataset.value, "threshold": given.threshold, "budget": given.budget, "calls": []}
    ledger = json.loads(path.read_text())
    if ledger["dataset"] != given.dataset.value:
        raise ValueError(f"this task's ledger is for dataset {ledger['dataset']}: pass dataset {ledger['dataset']}")
    if parse_decimal(ledger["threshold"]) != decimal(given.threshold, "threshold"):
        raise ValueError(f"this task's ledger is for threshold {ledger['threshold']!r}: pass that threshold")
    if ledger["budget"] != given.budget:
        raise ValueError(f"this task's ledger is for budget {ledger['budget']}: pass budget {ledger['budget']}")
    return ledger


def score_pairs(given: ScoreInput, workspace: Path) -> ScoreOutput:
    threshold = decimal(given.threshold, "threshold")
    if given.budget < 1:
        raise ValueError(f"budget must be at least 1, got {given.budget}: copy it from the task input")
    if given.start_rank < 1 or given.count < 1:
        raise ValueError(f"start_rank and count must each be at least 1, got {given.start_rank} and {given.count}")
    if not re.fullmatch(re.escape(given.dataset.value) + r"-[1-9][0-9]*x[1-9][0-9]*", given.run_id):
        raise ValueError(
            f"run_id {given.run_id!r} is not a {given.dataset.value} run:"
            f" pass a run_id lsh_candidates returned for it, {given.dataset.value}-<bits_per_band>x<bands>"
        )
    path = candidates_path(workspace, given.run_id)
    if not path.exists():
        raise ValueError(f"run {given.run_id} has no saved candidates: call lsh_candidates for it first")
    candidates = json.loads(path.read_text())["candidates"]
    first, last = given.start_rank, given.start_rank + given.count - 1
    if first > len(candidates):
        raise ValueError(f"run {given.run_id} has {len(candidates)} candidates: start_rank must be at most that")
    if last > len(candidates):
        raise ValueError(
            f"ranks {first} to {last} go past run {given.run_id}'s {len(candidates)} candidates:"
            f" with start_rank {first}, count may be at most {len(candidates) - first + 1}"
        )

    ledger = _ledger(given, workspace)
    seen = {(a, b) for call in ledger["calls"] for a, b in call["scored"]}
    remaining = given.budget - len(seen)
    span = [(a, b) for a, b, _ in candidates[first - 1 : last]]
    new = [pair for pair in span if pair not in seen]
    if len(new) > remaining:
        fits, fresh = 0, 0
        for pair in span:
            fresh += pair not in seen
            if fresh > remaining:
                break
            fits += 1
        raise ValueError(
            f"ranks {first} to {last} hold {len(new)} pairs not yet scored ({len(span) - len(new)} already scored),"
            f" but only {remaining} of the budget {given.budget} remain: with start_rank {first},"
            f" count may be at most {fits}" + ("; the budget is used up, so submit what you found" if not remaining else "")
        )

    data = load(given.dataset)
    qualifying = []
    for a, b in new:
        product = dot(data.vectors[a], data.vectors[b])
        if qualifies(product, data.norms[a], data.norms[b], threshold):
            qualifying.append(ScoredPair(a=a, b=b, dot=product, norm_a=data.norms[a], norm_b=data.norms[b]))
    ledger["calls"].append(
        {
            "run_id": given.run_id,
            "start_rank": first,
            "count": given.count,
            "scored": [list(pair) for pair in new],
            "skipped": len(span) - len(new),
            "qualifying": [
                {"a": p.a, "b": p.b, "dot": p.dot, "norm_a": p.norm_a, "norm_b": p.norm_b} for p in qualifying
            ],
        }
    )
    (workspace / LEDGER).write_text(json.dumps(ledger) + "\n")
    total = len(seen) + len(new)
    return ScoreOutput(
        qualifying=qualifying,
        scored=len(new),
        skipped=len(span) - len(new),
        total_scored=total,
        remaining=given.budget - total,
    )
