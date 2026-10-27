"""lsh_candidates: SimHash LSH candidate pairs, ranked by how many bands they share, saved in the workspace.

Each of the first bits_per_band x bands hyperplanes gives each keyword one sign bit (a zero dot product counts
as positive). Band k is bits k * bits_per_band to (k + 1) * bits_per_band. A pair is a candidate when the
signatures of at least one band match. The candidates go to candidates/<run_id>.json, most matched bands
first, ties in (a, b) order, so that they never pass through the model.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import combinations, groupby
from pathlib import Path

from contract import BandCount, LshInput, LshOutput
from plugin.bundle import candidates_path, canonical, dot, hyperplanes, load, run_id


def lsh_candidates(given: LshInput, workspace: Path) -> LshOutput:
    bits, bands = given.bits_per_band, given.bands
    planes = hyperplanes()
    if bits < 1 or bands < 1:
        raise ValueError(f"bits_per_band and bands must each be at least 1, got {bits} and {bands}")
    if bits * bands > len(planes):
        raise ValueError(
            f"bits_per_band x bands is {bits * bands}, but only {len(planes)} hyperplanes exist:"
            f" choose values whose product is at most {len(planes)}"
        )
    data = load(given.dataset)
    signs = {
        keyword: [dot(plane, data.vectors[keyword]) >= 0 for plane in planes[: bits * bands]]
        for keyword in data.keywords
    }
    matched: Counter[tuple[str, str]] = Counter()
    for band in range(bands):
        buckets = defaultdict(list)
        for keyword in data.keywords:
            buckets[tuple(signs[keyword][band * bits : (band + 1) * bits])].append(keyword)
        for bucket in buckets.values():
            matched.update(canonical(a, b) for a, b in combinations(bucket, 2))
    ranked = sorted(matched.items(), key=lambda item: (-item[1], item[0]))

    run = run_id(given.dataset, bits, bands)
    path = candidates_path(workspace, run)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = {
        "run_id": run,
        "dataset": given.dataset.value,
        "bits_per_band": bits,
        "bands": bands,
        "candidates": [[a, b, count] for (a, b), count in ranked],
    }
    path.write_text(json.dumps(saved) + "\n")

    histogram, rank = [], 1
    for count, group in groupby(ranked, key=lambda item: item[1]):
        size = sum(1 for _ in group)
        histogram.append(BandCount(matched_bands=count, candidates=size, first_rank=rank, last_rank=rank + size - 1))
        rank += size
    return LshOutput(run_id=run, candidates=len(ranked), histogram=histogram)
