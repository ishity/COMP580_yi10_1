"""The bundled datasets and hyperplanes, and the names and paths the tools share."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from pathlib import Path

from contract import Dataset
from plugin.rule import DECIMAL_RULE, parse_decimal

DATA = Path(__file__).with_name("data")
LEDGER = "score_ledger.json"
CANDIDATES = "candidates"


@dataclass(frozen=True)
class Bundle:
    keywords: tuple[str, ...]
    vectors: dict[str, tuple[int, ...]]
    norms: dict[str, int]


@cache
def load(dataset: Dataset) -> Bundle:
    """``dataset``'s keywords, in order, with their integer vectors and squared lengths."""
    raw = json.loads((DATA / f"{dataset.value}.json").read_text())
    keywords, vectors = raw["keywords"], raw["vectors"]
    if len(keywords) != len(vectors) or len(set(keywords)) != len(keywords):
        raise ValueError(f"{dataset.value} must hold one vector per keyword and unique keywords")
    if len({len(vector) for vector in vectors}) != 1:
        raise ValueError(f"{dataset.value}'s vectors must all have the same dimension")
    by_keyword = {keyword: tuple(vector) for keyword, vector in zip(keywords, vectors)}
    norms = {keyword: sum(x * x for x in vector) for keyword, vector in by_keyword.items()}
    if not all(norms.values()):
        raise ValueError(f"{dataset.value} holds a zero vector, whose cosine is undefined")
    return Bundle(keywords=tuple(keywords), vectors=by_keyword, norms=norms)


@cache
def hyperplanes() -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(plane) for plane in json.loads((DATA / "hyperplanes.json").read_text()))


def dot(u: tuple[int, ...], v: tuple[int, ...]) -> int:
    return sum(x * y for x, y in zip(u, v))


def canonical(a: str, b: str) -> tuple[str, str]:
    """A pair in canonical order: the keyword that sorts first, then the other."""
    return (a, b) if a < b else (b, a)


def decimal(text: str, name: str) -> Fraction:
    value = parse_decimal(text)
    if value is None:
        raise ValueError(f"{name} must be {DECIMAL_RULE}, got {text!r}")
    return value


def run_id(dataset: Dataset, bits_per_band: int, bands: int) -> str:
    return f"{dataset.value}-{bits_per_band}x{bands}"


def candidates_path(workspace: Path, run: str) -> Path:
    return workspace / CANDIDATES / f"{run}.json"
