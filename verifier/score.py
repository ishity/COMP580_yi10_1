from pathlib import Path

from .checks import (
    CANDIDATES,
    LEDGER,
    decimal_problem,
    qualifying_problem,
    read_json,
    result,
    run_id_problem,
    scored,
)
from .rule import parse_decimal


def verify_input(value):
    problem = decimal_problem(value.threshold, "threshold") or run_id_problem(value.dataset, value.run_id)
    if problem:
        return result(False, problem)
    if value.start_rank < 1 or value.count < 1:
        return result(False, "start_rank and count must each be at least 1")
    if value.budget < 1:
        return result(False, "budget must be at least 1")
    return result(True, "input is valid")


def verify_output(input_value, output_value, workspace):
    ledger = read_json(Path(workspace) / LEDGER)
    if ledger is None or not ledger["calls"]:
        return result(False, "score_pairs left no record of its call in the ledger")
    if (ledger["dataset"], parse_decimal(ledger["threshold"]), ledger["budget"]) != (
        input_value.dataset.value,
        parse_decimal(input_value.threshold),
        input_value.budget,
    ):
        return result(False, "the ledger is for another dataset, threshold or budget than this call")
    *earlier, call = ledger["calls"]
    if (call["run_id"], call["start_rank"], call["count"]) != (
        input_value.run_id,
        input_value.start_rank,
        input_value.count,
    ):
        return result(False, "the ledger's last call is not this call's run and rank range")
    run = read_json(Path(workspace) / CANDIDATES / f"{input_value.run_id}.json")
    if run is None:
        return result(False, f"run {input_value.run_id}'s candidates are not saved in the workspace")
    first = input_value.start_rank
    span = [(a, b) for a, b, _ in run["candidates"][first - 1 : first - 1 + input_value.count]]
    if len(span) != input_value.count:
        return result(False, "the rank range goes past the run's candidates")
    seen = {(a, b) for past in earlier for a, b in past["scored"]}
    new = [pair for pair in span if pair not in seen]
    if [tuple(pair) for pair in call["scored"]] != new or call["skipped"] != len(span) - len(new):
        return result(False, "the ledger does not score exactly the range's pairs no earlier call scored")
    if (output_value.scored, output_value.skipped) != (len(new), len(span) - len(new)):
        return result(False, f"of the range's {len(span)} pairs, {len(new)} are new: scored and skipped disagree")
    qualifying = [scored(pair) for pair in output_value.qualifying]
    if qualifying != call["qualifying"]:
        return result(False, "the qualifying pairs differ from the ledger's")
    order = {pair: rank for rank, pair in enumerate(new)}
    ranks = [order.get((pair["a"], pair["b"])) for pair in qualifying]
    if None in ranks or ranks != sorted(set(ranks)):
        return result(False, "the qualifying pairs are not newly scored pairs of the range, in rank order")
    threshold = parse_decimal(input_value.threshold)
    for pair in qualifying:
        problem = qualifying_problem(pair, threshold)
        if problem:
            return result(False, problem)
    total = len(seen) + len(new)
    if output_value.total_scored != total or output_value.remaining != input_value.budget - total:
        return result(False, f"{total} distinct pairs are scored: total_scored or remaining disagree")
    if total > input_value.budget:
        return result(False, f"{total} distinct pairs are scored, over the budget of {input_value.budget}")
    return result(True, "the range is scored within budget", scored=len(new), qualifying=len(qualifying))
