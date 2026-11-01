from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest
from slimconfig import build_output, plain_output

import verifier
from contract import TASKS, TOOLS, Dataset, ExactOutput, FindInput, LshInput, ScoreInput, Task, Tool
from plugin import exact_pairs, lsh_candidates, score_pairs
from plugin.ledgers import ledger_path

EDGE = {"dataset": "edge-cases", "threshold": "0.90", "budget": 190, "min_recall": "1.0"}
EMPTY = {"dataset": "nlp-small", "threshold": "0.90", "budget": 500, "min_recall": "1.0"}
SMALL = {"dataset": "nlp-small", "threshold": "0.80", "budget": 2000, "min_recall": "0.90"}
EXACT_COSINE = {"a": "base", "b": "cos-0.90-with-base"}


def ledger_of(workspace: Path, arguments: dict) -> Path:
    """The instance's ledger in ``workspace``."""
    return ledger_path(workspace, arguments["dataset"], Fraction(arguments["threshold"]), arguments["budget"])


# The verifier gets contract instances, as NanoAgent's sandbox builds them from JSON.
def verify_task(arguments: dict, pairs: list[dict], workspace: Path, task: dict | None = None) -> dict:
    """The verdict on ``pairs`` for the task ``arguments``, whose reference exact_pairs computes from ``task``
    (the same arguments unless given)."""
    given, made = TASKS[Task.FIND_SIMILAR_PAIRS]
    reference = plain_output(exact_pairs(build_output(task or arguments, given), workspace / "scratch"))
    return verifier.verify_task(
        Task.FIND_SIMILAR_PAIRS.value,
        build_output(arguments, given),
        build_output({"pairs": pairs}, made),
        build_output(reference, ExactOutput),
        str(workspace),
    )


def verify_input(tool: Tool, value: dict) -> dict:
    return verifier.verify_input(tool.value, build_output(value, TOOLS[tool][0]))


def reference(arguments: dict) -> list[dict]:
    found = exact_pairs(build_output(arguments, FindInput), Path("."))
    return [{"a": pair.a, "b": pair.b} for pair in found.pairs]


def scored(workspace: Path, arguments: dict, last_rank: int | None = None, bits: int = 1, bands: int = 64) -> list[dict]:
    """Run LSH, score ranks 1 to ``last_rank`` (every candidate when None) and return the qualifying pairs."""
    dataset = Dataset(arguments["dataset"])
    run = lsh_candidates(LshInput(dataset=dataset, bits_per_band=bits, bands=bands), workspace)
    made = score_pairs(
        ScoreInput(
            dataset=dataset,
            run_id=run.run_id,
            start_rank=1,
            count=run.candidates if last_rank is None else last_rank,
            threshold=arguments["threshold"],
            budget=arguments["budget"],
        ),
        workspace,
    )
    return [{"a": pair.a, "b": pair.b} for pair in made.qualifying + made.previously_qualifying]


@pytest.fixture
def edge(tmp_path: Path) -> list[dict]:
    """Every edge-cases candidate scored at 0.90 in ``tmp_path``; the qualifying pairs."""
    found = scored(tmp_path, EDGE)
    assert sorted(found, key=lambda pair: (pair["a"], pair["b"])) == reference(EDGE)
    return found


def test_approves_every_qualifying_pair(tmp_path: Path, edge: list[dict]) -> None:
    verdict = verify_task(EDGE, edge, tmp_path)
    assert verdict["valid"] is True and verdict["score"] == 1.0
    assert verdict["details"] == {"submitted": 22, "reference": 22, "scored": 186}


def test_the_verdict_is_the_same_twice(tmp_path: Path, edge: list[dict]) -> None:
    assert verify_task(EDGE, edge, tmp_path) == verify_task(EDGE, edge, tmp_path)


def test_approves_a_pair_whose_cosine_equals_the_threshold(tmp_path: Path, edge: list[dict]) -> None:
    assert EXACT_COSINE in edge
    assert verify_task(EDGE | {"min_recall": "0.001"}, [EXACT_COSINE], tmp_path)["valid"] is True


def test_refuses_one_wrong_pair_with_score_zero(tmp_path: Path, edge: list[dict]) -> None:
    wrong = {"a": "base", "b": "just-below-0.90-with-base"}
    verdict = verify_task(EDGE, [*edge, wrong], tmp_path)
    assert verdict["valid"] is False and verdict["score"] == 0.0 and "do not qualify" in verdict["message"]


def test_refuses_an_empty_answer_when_pairs_qualify(tmp_path: Path, edge: list[dict]) -> None:
    verdict = verify_task(EDGE, [], tmp_path)
    assert verdict["valid"] is False and verdict["score"] == 0.0 and "recall 0.000" in verdict["message"]


def test_approves_an_empty_answer_when_no_pair_qualifies(tmp_path: Path) -> None:
    assert reference(EMPTY) == []
    assert verify_task(EMPTY, [], tmp_path)["valid"] is True  # no ledger: nothing was scored
    assert scored(tmp_path, EMPTY, last_rank=500, bits=6, bands=16) == []
    verdict = verify_task(EMPTY, [], tmp_path)
    assert verdict["valid"] is True and verdict["score"] == 1.0 and verdict["details"]["scored"] == 500


def test_accepts_a_reversed_pair_but_not_a_repeated_one(tmp_path: Path, edge: list[dict]) -> None:
    reversed_pairs = [{"a": pair["b"], "b": pair["a"]} for pair in edge]
    assert verify_task(EDGE, reversed_pairs, tmp_path)["valid"] is True
    for repeat in (edge[0], reversed_pairs[0]):
        verdict = verify_task(EDGE, [*edge, repeat], tmp_path)
        assert verdict["valid"] is False and "submitted twice" in verdict["message"]


@pytest.mark.parametrize(
    ("pair", "message"),
    [
        ({"a": "base", "b": "no such keyword"}, "is not a keyword of edge-cases"),
        ({"a": "base", "b": "base"}, "with itself"),
    ],
)
def test_refuses_a_malformed_pair(tmp_path: Path, edge: list[dict], pair: dict, message: str) -> None:
    verdict = verify_task(EDGE, [*edge, pair], tmp_path)
    assert verdict["valid"] is False and message in verdict["message"]


def test_refuses_a_ledger_over_the_budget(tmp_path: Path, edge: list[dict]) -> None:
    ledger = json.loads(ledger_of(tmp_path, EDGE).read_text())
    ledger_of(tmp_path, EDGE | {"budget": 100}).write_text(json.dumps(ledger | {"budget": 100}))
    verdict = verify_task(EDGE | {"budget": 100}, edge, tmp_path, task=EDGE)
    assert verdict["valid"] is False and "186 distinct pairs were scored, over the budget of 100" in verdict["message"]


@pytest.mark.parametrize(
    "changes", [{"budget": 500}, {"threshold": "0.80"}, {"dataset": "nlp-small"}], ids=["budget", "threshold", "dataset"]
)
def test_refuses_a_ledger_that_records_another_task(tmp_path: Path, edge: list[dict], changes: dict) -> None:
    ledger = json.loads(ledger_of(tmp_path, EDGE).read_text())
    ledger_of(tmp_path, EDGE).write_text(json.dumps(ledger | changes))
    verdict = verify_task(EDGE, edge, tmp_path)
    assert verdict["valid"] is False and "the ledger records another" in verdict["message"]


def test_another_instance_does_not_see_the_ledger(tmp_path: Path, edge: list[dict]) -> None:
    for changes in ({"budget": 500}, {"threshold": "0.80"}):
        verdict = verify_task(EDGE | changes, edge[:1], tmp_path)
        assert verdict["valid"] is False and "there is no ledger edge-cases-" in verdict["message"]


def test_refuses_a_ledger_that_scores_a_pair_twice(tmp_path: Path, edge: list[dict]) -> None:
    ledger = json.loads(ledger_of(tmp_path, EDGE).read_text())
    ledger["calls"].append(ledger["calls"][0] | {"qualifying": []})
    ledger_of(tmp_path, EDGE).write_text(json.dumps(ledger))
    verdict = verify_task(EDGE, edge, tmp_path)
    assert verdict["valid"] is False and "scores a pair twice" in verdict["message"]


def test_refuses_pairs_without_a_ledger(tmp_path: Path, edge: list[dict]) -> None:
    ledger_of(tmp_path, EDGE).unlink()
    verdict = verify_task(EDGE, edge, tmp_path)
    assert verdict["valid"] is False and "no ledger edge-cases-0.900-190.json" in verdict["message"]


def test_refuses_a_qualifying_pair_that_was_never_scored(tmp_path: Path) -> None:
    run = lsh_candidates(LshInput(dataset=Dataset.EDGE_CASES, bits_per_band=1, bands=64), tmp_path)
    ranks = {(a, b): rank for rank, (a, b, _) in enumerate(
        json.loads((tmp_path / "candidates" / f"{run.run_id}.json").read_text())["candidates"], 1
    )}
    pairs = reference(EDGE)
    last = max(ranks[(pair["a"], pair["b"])] for pair in pairs)
    found = scored(tmp_path, EDGE, last_rank=last - 1)
    assert len(found) == len(pairs) - 1
    verdict = verify_task(EDGE, pairs, tmp_path)
    assert verdict["valid"] is False and "never returned as qualifying" in verdict["message"]
    assert verify_task(EDGE | {"min_recall": "0.95"}, found, tmp_path)["valid"] is True


def test_refuses_a_pair_recorded_with_other_integers(tmp_path: Path, edge: list[dict]) -> None:
    ledger = json.loads(ledger_of(tmp_path, EDGE).read_text())
    ledger["calls"][0]["qualifying"][0]["norm_a"] -= 1
    ledger_of(tmp_path, EDGE).write_text(json.dumps(ledger))
    verdict = verify_task(EDGE, edge, tmp_path)
    assert verdict["valid"] is False and "other integers" in verdict["message"]


def test_recall_exactly_at_min_recall_passes(tmp_path: Path, edge: list[dict]) -> None:
    half = EDGE | {"min_recall": "0.5"}
    verdict = verify_task(half, edge[:11], tmp_path)
    assert verdict["valid"] is True and verdict["score"] == 0.5
    verdict = verify_task(half, edge[:10], tmp_path)
    assert verdict["valid"] is False and verdict["score"] == 10 / 22


@pytest.mark.parametrize(
    ("tool", "value", "valid"),
    [
        (Tool.EXACT_PAIRS, EDGE, True),
        (Tool.EXACT_PAIRS, EDGE | {"threshold": "1"}, True),
        (Tool.EXACT_PAIRS, EDGE | {"threshold": "0.8000"}, False),
        (Tool.EXACT_PAIRS, EDGE | {"threshold": "1.5"}, False),
        (Tool.EXACT_PAIRS, EDGE | {"threshold": "0"}, False),
        (Tool.EXACT_PAIRS, EDGE | {"min_recall": "0.9e0"}, False),
        (Tool.EXACT_PAIRS, EDGE | {"budget": 0}, False),
        (Tool.LSH_CANDIDATES, {"dataset": "nlp-large", "bits_per_band": 8, "bands": 32}, True),
        (Tool.LSH_CANDIDATES, {"dataset": "nlp-large", "bits_per_band": 9, "bands": 32}, False),
        (Tool.LSH_CANDIDATES, {"dataset": "nlp-large", "bits_per_band": 0, "bands": 32}, False),
    ],
)
def test_checks_what_the_types_cannot_say(tool: Tool, value: dict, valid: bool) -> None:
    assert verify_input(tool, value)["valid"] is valid


@pytest.mark.parametrize(
    ("changes", "valid"),
    [({}, True), ({"run_id": "nlp-large-6x32"}, False), ({"run_id": "nlp-small-6"}, False), ({"count": 0}, False)],
)
def test_checks_a_score_call(changes: dict, valid: bool) -> None:
    value = {"dataset": "nlp-small", "run_id": "nlp-small-6x32", "start_rank": 1, "count": 100}
    assert verify_input(Tool.SCORE_PAIRS, value | {"threshold": "0.80", "budget": 2000} | changes)["valid"] is valid


# NanoAgent copies a member's workspace files into every run and back, so one workspace sees many runs.
def test_two_instances_share_a_workspace(tmp_path: Path, edge: list[dict]) -> None:
    assert scored(tmp_path, EDGE)[0] in edge  # edge-cases' budget is used up, yet nlp-small is not locked
    found = scored(tmp_path, SMALL, last_rank=SMALL["budget"], bits=6, bands=16)
    assert verify_task(SMALL, found, tmp_path)["valid"] is True
    assert verify_task(EDGE, edge, tmp_path)["valid"] is True
    assert sorted(path.name for path in (tmp_path / "ledgers").iterdir()) == [
        "edge-cases-0.900-190.json",
        "nlp-small-0.800-2000.json",
    ]


def test_the_same_instance_passes_when_run_again(tmp_path: Path, edge: list[dict]) -> None:
    assert verify_task(EDGE, edge, tmp_path)["valid"] is True
    rerun = scored(tmp_path, EDGE)  # every pair was scored by the first run: all skipped, none charged
    assert sorted(rerun, key=lambda pair: (pair["a"], pair["b"])) == reference(EDGE)
    verdict = verify_task(EDGE, rerun, tmp_path)
    assert verdict["valid"] is True and verdict["score"] == 1.0
    assert verdict["details"] == {"submitted": 22, "reference": 22, "scored": 186}
