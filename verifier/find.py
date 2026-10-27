"""The find-similar-pairs task's output check, against the exact_pairs reference and the score ledger.

The first failing check decides: malformed pairs, the budget, precision, provenance, then recall.
"""

from fractions import Fraction
from pathlib import Path

from .checks import LEDGER, qualifying_problem, read_json, result, scored
from .rule import parse_decimal


def verify_task(arguments, output, reference, workspace):
    """``arguments``: a contract FindInput; ``output``: a FindOutput; ``reference``: exact_pairs' ExactOutput."""
    if (reference.dataset, reference.threshold) != (arguments.dataset, arguments.threshold):
        return result(False, "the reference is for another dataset or threshold than the task")
    threshold, min_recall = parse_decimal(arguments.threshold), parse_decimal(arguments.min_recall)
    if threshold is None or min_recall is None:
        return result(False, "threshold and min_recall must be decimal strings in (0, 1]")

    # 1. Malformed: two distinct known keywords per pair, each pair once in either order.
    keywords = set(reference.keywords)
    submitted = []
    for pair in output.pairs:
        if pair.a == pair.b:
            return result(False, f"pair ({pair.a!r}, {pair.b!r}) pairs a keyword with itself")
        for keyword in (pair.a, pair.b):
            if keyword not in keywords:
                return result(False, f"{keyword!r} is not a keyword of {arguments.dataset.value}")
        submitted.append((min(pair.a, pair.b), max(pair.a, pair.b)))
    if len(set(submitted)) != len(submitted):
        return result(False, "a pair is submitted twice, counting (a, b) and (b, a) as one pair")

    # 2. Budget: the ledger is this task's, and its distinct scored pairs are within the budget.
    ledger = read_json(Path(workspace) / LEDGER)
    calls = [] if ledger is None else ledger["calls"]
    if ledger is None and submitted:
        return result(False, f"there is no {LEDGER}, so no submitted pair was scored")
    if ledger is not None and (ledger["dataset"], parse_decimal(ledger["threshold"]), ledger["budget"]) != (
        arguments.dataset.value,
        threshold,
        arguments.budget,
    ):
        return result(False, "the ledger is for another dataset, threshold or budget than the task")
    pairs_scored = [tuple(pair) for call in calls for pair in call["scored"]]
    if len(set(pairs_scored)) != len(pairs_scored):
        return result(False, "the ledger scores a pair twice")
    if len(pairs_scored) > arguments.budget:
        return result(False, f"{len(pairs_scored)} distinct pairs were scored, over the budget of {arguments.budget}")

    # 3. Precision: every submitted pair qualifies by the reference.
    truth = {(pair.a, pair.b): scored(pair) for pair in reference.pairs}
    false = [pair for pair in submitted if pair not in truth]
    if false:
        return result(False, f"{len(false)} submitted pairs do not qualify, e.g. {false[0]}", false_pairs=len(false))

    # 4. Provenance: score_pairs returned every submitted pair as qualifying, with the reference's integers.
    returned = {(pair["a"], pair["b"]): pair for call in calls for pair in call["qualifying"]}
    for pair in submitted:
        if pair not in returned:
            return result(False, f"pair {pair} was never returned as qualifying by score_pairs")
        if returned[pair] != truth[pair] or qualifying_problem(returned[pair], threshold):
            return result(False, f"pair {pair} was recorded with other integers than the dataset's")

    # 5. Recall.
    recall = Fraction(len(submitted), len(truth)) if truth else Fraction(1)
    details = {"submitted": len(submitted), "reference": len(truth), "scored": len(pairs_scored)}
    message = f"found {len(submitted)} of {len(truth)} qualifying pairs, scoring {len(pairs_scored)} pairs"
    if recall < min_recall:
        return result(False, f"{message}: recall {float(recall):.3f} is below {arguments.min_recall}", score=float(recall), **details)
    return result(True, message, score=float(recall), **details)
