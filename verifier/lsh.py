from itertools import groupby
from pathlib import Path

from .checks import CANDIDATES, HYPERPLANES, read_json, result


def verify_input(value):
    if value.bits_per_band < 1 or value.bands < 1:
        return result(False, "bits_per_band and bands must each be at least 1")
    if value.bits_per_band * value.bands > HYPERPLANES:
        return result(False, f"bits_per_band x bands must be at most {HYPERPLANES}, the hyperplanes there are")
    return result(True, "input is valid")


def candidates_problem(run, input_value):
    """What is wrong with a saved run's ranked candidates, or None."""
    if (run["dataset"], run["bits_per_band"], run["bands"]) != (
        input_value.dataset.value,
        input_value.bits_per_band,
        input_value.bands,
    ):
        return "the saved run is not for the requested dataset, bits_per_band and bands"
    pairs = [(a, b) for a, b, _ in run["candidates"]]
    if len(set(pairs)) != len(pairs):
        return "the saved run lists a candidate pair twice"
    for a, b, matched in run["candidates"]:
        if not a < b:
            return f"candidate ({a!r}, {b!r}) is not in canonical order, a before b"
        if not 1 <= matched <= input_value.bands:
            return f"candidate ({a!r}, {b!r}) matches {matched} bands, not 1 to {input_value.bands}"
    keys = [(-matched, a, b) for a, b, matched in run["candidates"]]
    if keys != sorted(keys):
        return "the saved candidates are not ranked by matched bands, most first, then by (a, b)"
    return None


def verify_output(input_value, output_value, workspace):
    expected = f"{input_value.dataset.value}-{input_value.bits_per_band}x{input_value.bands}"
    if output_value.run_id != expected:
        return result(False, f"run_id must be {expected}, not {output_value.run_id}")
    run = read_json(Path(workspace) / CANDIDATES / f"{expected}.json")
    if run is None or run["run_id"] != expected:
        return result(False, f"run {expected}'s candidates are not saved in the workspace")
    problem = candidates_problem(run, input_value)
    if problem:
        return result(False, problem)
    if output_value.candidates != len(run["candidates"]):
        return result(False, f"output counts {output_value.candidates} candidates, the run saved {len(run['candidates'])}")
    histogram, rank = [], 1
    for matched, group in groupby(run["candidates"], key=lambda candidate: candidate[2]):
        size = sum(1 for _ in group)
        histogram.append((matched, size, rank, rank + size - 1))
        rank += size
    given = [(row.matched_bands, row.candidates, row.first_rank, row.last_rank) for row in output_value.histogram]
    if given != histogram:
        return result(False, "the histogram does not match the saved candidates")
    return result(True, "the candidates are ranked and counted", candidates=output_value.candidates)
