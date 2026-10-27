"""describe_dataset: how big a bundled dataset is, and how many hyperplanes LSH may use."""

from __future__ import annotations

from pathlib import Path

from contract import DescribeInput, DescribeOutput
from plugin.bundle import hyperplanes, load


def describe_dataset(given: DescribeInput, workspace: Path) -> DescribeOutput:
    data = load(given.dataset)
    count = len(data.keywords)
    return DescribeOutput(
        dataset=given.dataset,
        keywords=count,
        dimension=len(data.vectors[data.keywords[0]]),
        hyperplanes=len(hyperplanes()),
        pairs=count * (count - 1) // 2,
    )
