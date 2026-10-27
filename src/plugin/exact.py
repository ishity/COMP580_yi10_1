"""exact_pairs: every qualifying pair of a dataset by brute force, the task's reference."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

from contract import ExactOutput, FindInput, ScoredPair
from plugin.bundle import canonical, decimal, dot, load
from plugin.rule import qualifies


def exact_pairs(given: FindInput, workspace: Path) -> ExactOutput:
    threshold = decimal(given.threshold, "threshold")
    data = load(given.dataset)
    pairs = []
    for x, y in combinations(data.keywords, 2):
        a, b = canonical(x, y)
        product = dot(data.vectors[a], data.vectors[b])
        if qualifies(product, data.norms[a], data.norms[b], threshold):
            pairs.append(ScoredPair(a=a, b=b, dot=product, norm_a=data.norms[a], norm_b=data.norms[b]))
    return ExactOutput(
        dataset=given.dataset,
        threshold=given.threshold,
        keywords=list(data.keywords),
        pairs=sorted(pairs, key=lambda pair: (pair.a, pair.b)),
    )
