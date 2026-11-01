from __future__ import annotations

import json
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import pytest
from slimconfig import build_output, plain_output

import verifier
import verifier.ledgers
import verifier.rule
from contract import TOOLS, Dataset, DescribeInput, FindInput, LshInput, ScoreInput, Tool
from plugin import describe_dataset, exact_pairs, lsh_candidates, rule, score_pairs
from plugin.bundle import hyperplanes, load
from plugin.ledgers import ledger_path

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = ["0.001", "0.5", "0.6", "0.7", "0.70", "0.707", "0.708", "0.8", "0.80", "0.801", "0.9", "0.90", "0.96", "1"]


def checked(tool: Tool, given, made, workspace: Path) -> dict:
    """verify_output on a tool call, with values built the way the sandbox builds them."""
    given_class, made_class = TOOLS[tool]
    return verifier.verify_output(
        tool.value,
        build_output(plain_output(given), given_class),
        build_output(plain_output(made), made_class),
        str(workspace),
    )


def test_describe_dataset_counts_the_bundle(tmp_path: Path) -> None:
    given = DescribeInput(dataset=Dataset.NLP_LARGE)
    described = describe_dataset(given, tmp_path)
    assert (described.keywords, described.dimension, described.hyperplanes, described.pairs) == (493, 32, 256, 121278)
    assert checked(Tool.DESCRIBE_DATASET, given, described, tmp_path)["valid"] is True
    assert len(hyperplanes()) == verifier.checks.HYPERPLANES


def test_lsh_candidates_is_deterministic_and_ranked(tmp_path: Path) -> None:
    given = LshInput(dataset=Dataset.NLP_SMALL, bits_per_band=6, bands=16)
    first = lsh_candidates(given, tmp_path / "one")
    second = lsh_candidates(given, tmp_path / "two")
    assert first == second and first.run_id == "nlp-small-6x16"
    saved = (tmp_path / "one" / "candidates" / "nlp-small-6x16.json").read_text()
    assert saved == (tmp_path / "two" / "candidates" / "nlp-small-6x16.json").read_text()
    assert sum(row.candidates for row in first.histogram) == first.candidates
    assert first.histogram[0].first_rank == 1 and first.histogram[-1].last_rank == first.candidates
    assert checked(Tool.LSH_CANDIDATES, given, first, tmp_path / "one")["valid"] is True
    run = json.loads(saved)
    run["candidates"].reverse()
    (tmp_path / "one" / "candidates" / "nlp-small-6x16.json").write_text(json.dumps(run))
    assert checked(Tool.LSH_CANDIDATES, given, first, tmp_path / "one")["valid"] is False


@pytest.mark.parametrize(("bits", "bands", "message"), [(0, 4, "at least 1"), (4, 0, "at least 1"), (9, 29, "at most 256")])
def test_lsh_candidates_refuses_impossible_parameters(tmp_path: Path, bits: int, bands: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        lsh_candidates(LshInput(dataset=Dataset.EDGE_CASES, bits_per_band=bits, bands=bands), tmp_path)


@pytest.mark.parametrize("module", ["rule.py", "ledgers.py"])
def test_the_shared_files_are_identical(module: str) -> None:
    assert (ROOT / "src/plugin" / module).read_text() == (ROOT / "verifier" / module).read_text()


def test_tool_and_verifier_rules_agree_on_every_edge_case_pair() -> None:
    data = load(Dataset.EDGE_CASES)
    for text in THRESHOLDS:
        assert rule.parse_decimal(text) == verifier.rule.parse_decimal(text) == Fraction(text)
        threshold = Fraction(text)
        for a, b in combinations(data.keywords, 2):
            dot = sum(x * y for x, y in zip(data.vectors[a], data.vectors[b]))
            assert rule.qualifies(dot, data.norms[a], data.norms[b], threshold) == verifier.rule.qualifies(
                dot, data.norms[a], data.norms[b], threshold
            )


@pytest.mark.parametrize("text", ["0", "0.0", "1.001", "1.5", "0.8000", ".8", "0.8e0", " 0.8", "-0.5", "0,8", "nan", ""])
def test_decimals_outside_the_rule_are_refused(text: str) -> None:
    assert rule.parse_decimal(text) is None and verifier.rule.parse_decimal(text) is None


def exact_set(dataset: Dataset, threshold: str) -> set[tuple[str, str]]:
    given = FindInput(dataset=dataset, threshold=threshold, budget=1, min_recall="1")
    return {(pair.a, pair.b) for pair in exact_pairs(given, Path(".")).pairs}


def test_exact_cosines_sit_on_the_boundary() -> None:
    at_90 = exact_set(Dataset.EDGE_CASES, "0.90")
    assert ("base", "cos-0.90-with-base") in at_90
    assert ("base", "just-below-0.90-with-base") not in at_90
    assert ("base", "cos-0.80-with-base") in exact_set(Dataset.EDGE_CASES, "0.80")
    assert ("base", "cos-0.80-with-base") not in exact_set(Dataset.EDGE_CASES, "0.801")
    assert ("base", "cos-0.70-with-base") in exact_set(Dataset.EDGE_CASES, "0.7")
    assert ("base", "opposite-of-base") not in exact_set(Dataset.EDGE_CASES, "0.001")
    assert ("base", "orthogonal-to-base") not in exact_set(Dataset.EDGE_CASES, "0.001")
    assert ("base", "base-same-direction") in exact_set(Dataset.EDGE_CASES, "1")


def slow_exact(dataset: Dataset, threshold: str) -> set[tuple[str, str]]:
    """A second brute force: the squared cosine as one Fraction, from the raw JSON."""
    raw = json.loads((ROOT / "src/plugin/data" / f"{dataset.value}.json").read_text())
    pairs = set()
    for (a, u), (b, v) in combinations(zip(raw["keywords"], raw["vectors"]), 2):
        dot = sum(x * y for x, y in zip(u, v))
        if dot >= 0 and Fraction(dot * dot, sum(x * x for x in u) * sum(y * y for y in v)) >= Fraction(threshold) ** 2:
            pairs.add((min(a, b), max(a, b)))
    return pairs


@pytest.mark.parametrize(("threshold", "count"), [("0.80", 21), ("0.90", 0), ("0.70", 89)])
def test_exact_pairs_matches_a_slow_reference(tmp_path: Path, threshold: str, count: int) -> None:
    given = FindInput(dataset=Dataset.NLP_SMALL, threshold=threshold, budget=2000, min_recall="0.90")
    found = exact_pairs(given, tmp_path)
    assert {(pair.a, pair.b) for pair in found.pairs} == slow_exact(Dataset.NLP_SMALL, threshold)
    assert len(found.pairs) == count
    assert checked(Tool.EXACT_PAIRS, given, found, tmp_path)["valid"] is True


def score(workspace: Path, start: int, count: int, budget: int = 10, threshold: str = "0.90", **changes):
    fields = {"run_id": "edge-cases-1x64", "start_rank": start, "count": count, "threshold": threshold, "budget": budget}
    given = ScoreInput(dataset=Dataset.EDGE_CASES, **(fields | changes))
    return given, score_pairs(given, workspace)


@pytest.fixture
def run(tmp_path: Path) -> Path:
    lsh_candidates(LshInput(dataset=Dataset.EDGE_CASES, bits_per_band=1, bands=64), tmp_path)
    return tmp_path


def test_score_pairs_refuses_a_range_over_budget_and_says_what_fits(run: Path) -> None:
    with pytest.raises(ValueError, match=r"hold 20 pairs not yet scored \(0 already scored\), but only 10 of the budget"
                       r" 10 remain: with start_rank 1, count may be at most 10"):
        score(run, 1, 20)
    assert not (run / "ledgers").exists()
    given, made = score(run, 1, 10)
    assert (made.scored, made.skipped, made.total_scored, made.remaining) == (10, 0, 10, 0)
    assert checked(Tool.SCORE_PAIRS, given, made, run)["valid"] is True
    given, made = score(run, 5, 6)
    assert (made.scored, made.skipped, made.qualifying) == (0, 6, [])
    assert checked(Tool.SCORE_PAIRS, given, made, run)["valid"] is True
    with pytest.raises(ValueError, match="count may be at most 6; the budget is used up"):
        score(run, 5, 7)


def test_score_pairs_charges_only_new_pairs(run: Path) -> None:
    score(run, 1, 4)
    given, made = score(run, 3, 5)
    assert (made.scored, made.skipped, made.total_scored, made.remaining) == (3, 2, 7, 3)
    assert checked(Tool.SCORE_PAIRS, given, made, run)["valid"] is True
    ledger = json.loads(ledger_path(run, "edge-cases", Fraction("0.9"), 10).read_text())
    assert [len(call["scored"]) for call in ledger["calls"]] == [4, 3]


def test_each_instance_has_its_own_ledger(run: Path) -> None:
    score(run, 1, 2, threshold="0.9")
    _, made = score(run, 3, 2, threshold="0.90")
    assert made.total_scored == 4  # 0.9 and 0.90 are one threshold, so one ledger
    _, made = score(run, 1, 2, budget=11)
    assert (made.scored, made.total_scored) == (2, 2)
    _, made = score(run, 1, 2, threshold="0.8")
    assert (made.scored, made.total_scored) == (2, 2)
    names = sorted(path.name for path in (run / "ledgers").iterdir())
    assert names == ["edge-cases-0.800-10.json", "edge-cases-0.900-10.json", "edge-cases-0.900-11.json"]


@pytest.mark.parametrize(
    ("dataset", "threshold", "budget", "name"),
    [("nlp-large", "0.8", 3000, "nlp-large-0.800-3000.json"), ("edge-cases", "1", 190, "edge-cases-1.000-190.json"),
     ("nlp-small", "0.705", 1, "nlp-small-0.705-1.json")],
)
def test_ledger_names_write_the_threshold_to_3_places(dataset: str, threshold: str, budget: int, name: str) -> None:
    assert ledger_path(Path("w"), dataset, Fraction(threshold), budget) == Path("w/ledgers") / name
    assert verifier.ledgers.ledger_path(Path("w"), dataset, Fraction(threshold), budget) == Path("w/ledgers") / name


def test_score_pairs_refuses_a_ledger_whose_contents_contradict_its_name(run: Path) -> None:
    score(run, 1, 2)
    path = ledger_path(run, "edge-cases", Fraction("0.9"), 10)
    path.write_text(json.dumps(json.loads(path.read_text()) | {"budget": 11}))
    with pytest.raises(ValueError, match="records another dataset, threshold or budget than its name says"):
        score(run, 3, 1)


def test_a_rerun_gets_back_the_qualifying_pairs_it_skips(run: Path) -> None:
    given, first = score(run, 1, 30, budget=190)
    assert first.qualifying and first.previously_qualifying == []
    lsh_candidates(LshInput(dataset=Dataset.EDGE_CASES, bits_per_band=1, bands=64), run)  # the rerun's own call
    given, again = score(run, 1, 40, budget=190)
    assert (again.scored, again.skipped) == (10, 30)
    assert again.previously_qualifying == first.qualifying  # the same pairs, with the same integers
    assert checked(Tool.SCORE_PAIRS, given, again, run)["valid"] is True
    again.previously_qualifying.pop()
    assert checked(Tool.SCORE_PAIRS, given, again, run)["valid"] is False


def test_a_refusal_names_the_runs_already_scored(run: Path) -> None:
    score(run, 1, 10)
    with pytest.raises(ValueError, match="the budget is used up; ranges already scored, in runs edge-cases-1x64, cost"):
        score(run, 5, 7)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"run_id": "edge-cases-2x2"}, "call lsh_candidates for it first"),
        ({"run_id": "nlp-small-1x64"}, "is not a edge-cases run"),
        ({"run_id": "../x"}, "is not a edge-cases run"),
        ({"start_rank": 0}, "at least 1"),
        ({"threshold": "0.8000"}, "at most 3 decimal places"),
        ({"budget": 0}, "budget must be at least 1"),
        ({"start_rank": 1000}, "start_rank must be at most"),
    ],
)
def test_score_pairs_says_what_is_wrong(run: Path, changes: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        score(run, 1, 1, **changes)


def test_score_pairs_says_how_far_a_run_goes(run: Path) -> None:
    candidates = len(json.loads((run / "candidates" / "edge-cases-1x64.json").read_text())["candidates"])
    with pytest.raises(ValueError, match=f"with start_rank 2, count may be at most {candidates - 1}"):
        score(run, 2, candidates, budget=500)


def test_verify_output_refuses_a_tampered_score(run: Path) -> None:
    given, made = score(run, 1, 30, budget=190)
    assert made.qualifying
    assert checked(Tool.SCORE_PAIRS, given, made, run)["valid"] is True
    made.remaining += 1
    assert checked(Tool.SCORE_PAIRS, given, made, run)["valid"] is False
    made.remaining -= 1
    made.qualifying[0].dot -= 1
    assert checked(Tool.SCORE_PAIRS, given, made, run)["valid"] is False
