"""Each tool's and task's typed input and output (see nanoagent.plugin.v0001.contract): declarations only."""

from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from slimconfig import Output


class Tool(Enum):
    DESCRIBE_DATASET = "describe_dataset"
    LSH_CANDIDATES = "lsh_candidates"
    SCORE_PAIRS = "score_pairs"
    EXACT_PAIRS = "exact_pairs"


class Task(Enum):
    FIND_SIMILAR_PAIRS = "find-similar-pairs"


class Dataset(Enum):
    NLP_SMALL = "nlp-small"
    NLP_LARGE = "nlp-large"
    EDGE_CASES = "edge-cases"


DATASET = "The bundled keyword dataset"
THRESHOLD = "The cosine similarity threshold as a decimal string, e.g. 0.80: a pair qualifies when its cosine is at least it"
BUDGET = "The most distinct pairs score_pairs may score in this task, over all its calls"
RUN_ID = "The LSH run's id, <dataset>-<bits_per_band>x<bands>, as lsh_candidates returned it"
KEYWORD_A = "The pair's first keyword, the one that sorts first"
KEYWORD_B = "The pair's second keyword, the one that sorts second"


@dataclass
class Pair(Output):
    a: Annotated[str, KEYWORD_A]
    b: Annotated[str, KEYWORD_B]


@dataclass
class ScoredPair(Output):
    a: Annotated[str, KEYWORD_A]
    b: Annotated[str, KEYWORD_B]
    dot: Annotated[int, "The dot product of the two keywords' integer vectors"]
    norm_a: Annotated[int, "The squared length of a's integer vector, the sum of its squared entries"]
    norm_b: Annotated[int, "The squared length of b's integer vector, the sum of its squared entries"]


@dataclass
class BandCount(Output):
    matched_bands: Annotated[int, "How many bands' signatures the candidates in this group share"]
    candidates: Annotated[int, "How many candidate pairs share exactly that many bands"]
    first_rank: Annotated[int, "The rank of this group's first candidate, from 1"]
    last_rank: Annotated[int, "The rank of this group's last candidate"]


@dataclass
class DescribeInput(Output):
    dataset: Annotated[Dataset, DATASET]


@dataclass
class DescribeOutput(Output):
    dataset: Annotated[Dataset, DATASET]
    keywords: Annotated[int, "How many keywords the dataset holds"]
    dimension: Annotated[int, "How many entries each keyword's integer vector has"]
    hyperplanes: Annotated[int, "How many fixed hyperplanes LSH may use, bits_per_band times bands at most"]
    pairs: Annotated[int, "How many keyword pairs there are, keywords choose 2"]


@dataclass
class LshInput(Output):
    dataset: Annotated[Dataset, DATASET]
    bits_per_band: Annotated[int, "How many hyperplane sign bits make one band's signature, at least 1"]
    bands: Annotated[int, "How many bands to hash, at least 1; a pair is a candidate when any band's signatures match"]


@dataclass
class LshOutput(Output):
    run_id: Annotated[str, RUN_ID]
    candidates: Annotated[int, "How many distinct candidate pairs the run found"]
    histogram: Annotated[
        list[BandCount], "The candidates grouped by how many bands they share, most first, in rank order"
    ]


@dataclass
class ScoreInput(Output):
    dataset: Annotated[Dataset, DATASET]
    run_id: Annotated[str, RUN_ID]
    start_rank: Annotated[int, "The rank of the first candidate to score, from 1"]
    count: Annotated[int, "How many candidates to score from start_rank on, at least 1"]
    threshold: Annotated[str, THRESHOLD]
    budget: Annotated[int, "The task input's budget, copied as given: the most distinct pairs score_pairs may score"]


@dataclass
class ScoreOutput(Output):
    qualifying: Annotated[list[ScoredPair], "The newly scored pairs in the range that qualify, in rank order"]
    scored: Annotated[int, "How many pairs in the range were scored now and charged to the budget"]
    skipped: Annotated[int, "How many pairs in the range were scored by an earlier call, so not scored or charged again"]
    total_scored: Annotated[int, "How many distinct pairs all calls so far have scored"]
    remaining: Annotated[int, "How many more pairs the budget allows"]


@dataclass
class FindInput(Output):
    dataset: Annotated[Dataset, DATASET]
    threshold: Annotated[str, THRESHOLD]
    budget: Annotated[int, BUDGET]
    min_recall: Annotated[str, "The least share of all qualifying pairs to find, as a decimal string, e.g. 0.90"]


@dataclass
class ExactOutput(Output):
    dataset: Annotated[Dataset, DATASET]
    threshold: Annotated[str, THRESHOLD]
    keywords: Annotated[list[str], "Every keyword in the dataset, in its order"]
    pairs: Annotated[list[ScoredPair], "Every qualifying pair, by brute force, sorted by a then b"]


@dataclass
class FindOutput(Output):
    pairs: Annotated[list[Pair], "Every qualifying pair score_pairs returned, each once"]


TOOLS = {
    Tool.DESCRIBE_DATASET: (DescribeInput, DescribeOutput),
    Tool.LSH_CANDIDATES: (LshInput, LshOutput),
    Tool.SCORE_PAIRS: (ScoreInput, ScoreOutput),
    Tool.EXACT_PAIRS: (FindInput, ExactOutput),
}
TASKS = {Task.FIND_SIMILAR_PAIRS: (FindInput, FindOutput)}
