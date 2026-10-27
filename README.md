# LSH Similar Pairs

A NanoAgent plugin with one checkable job: given a bundled keyword dataset with embeddings and a
similarity threshold τ, find every keyword pair whose cosine similarity is at least τ, using SimHash
LSH to choose which pairs to score, without scoring more pairs than a budget allows. The plugin is a
standard MCP server over stdio; NanoAgent owns its launch command, environment, input and output
checks and lifecycle.

## Tools

| Tool | Does | Code |
|---|---|---|
| `describe_dataset` | Gives a dataset's keyword count, vector dimension, the hyperplanes LSH may use (256) and its pair count | `src/plugin/describe.py` |
| `lsh_candidates` | Hashes every keyword with the first `bits_per_band × bands` hyperplanes, saves the candidate pairs (those sharing at least one band's signature) to `candidates/<run_id>.json` in the workspace, ranked by how many bands they share, and returns the run id, the count and a histogram by matched bands | `src/plugin/lsh.py` |
| `score_pairs` | Scores a rank range of a saved run with the exact rule and returns the qualifying pairs with their integer dot product and squared lengths; records every call in `score_ledger.json` and refuses a range that would take the distinct scored pairs past the budget | `src/plugin/score.py` |
| `exact_pairs` | Finds every qualifying pair by brute force: the task's reference, which the model never sees | `src/plugin/exact.py` |

### Tasks

| Task | Does | Runner |
|---|---|---|
| `find-similar-pairs` | Finds every keyword pair at least as similar as the threshold, within a scoring budget | `agent`, with `describe_dataset`, `lsh_candidates` and `score_pairs`; reference `exact_pairs` |

The task input is a `dataset`, a `threshold` and a `min_recall` (decimal strings such as `"0.80"`,
in (0, 1] with at most 3 decimal places), and a `budget` (the most distinct pairs `score_pairs` may
score). The output is the list of qualifying pairs.

A pair of integer vectors `u`, `v` qualifies when `dot = Σ uᵢvᵢ ≥ 0` and `dot² ≥ τ² · Σ uᵢ² · Σ vᵢ²`,
compared as fractions: no floating point decides a pair. `src/plugin/rule.py` and `verifier/rule.py`
hold the same rule, word for word.

Candidate lists never pass through the model: a wide LSH run on `nlp-large` has tens of thousands of
pairs. The agent sees only each run's histogram and asks `score_pairs` for rank ranges. The first
`score_pairs` call fixes the ledger's dataset, threshold and budget. A pair any earlier call scored is
skipped, not charged again, so the budget counts distinct pairs and a second LSH run costs only its
new pairs. Every refused call says what was wrong and which value would be valid, for example the
largest `count` the remaining budget allows.

`verify_task` checks, in order, and the first failure decides:

1. Malformed: every pair names two distinct keywords of the dataset, and no pair is submitted twice,
   counting `(a, b)` and `(b, a)` as one.
2. Budget: the ledger is for the task's dataset, threshold and budget, and its distinct scored pairs
   are within the budget. A non-empty answer without a ledger fails.
3. Precision: every submitted pair is in the reference. Any false pair scores 0.
4. Provenance: `score_pairs` returned every submitted pair as qualifying, with the reference's integers.
5. Recall: `recall = |submitted| / |reference|` (1 for an empty answer to an empty reference). The
   result is valid when recall is at least `min_recall`, and its score is the recall.

The bundled instances:

| Dataset | Keywords | Pairs | τ | Qualifying | Budget | min_recall |
|---|---|---|---|---|---|---|
| `nlp-small` | 200 | 19,900 | 0.80 | 21 | 2,000 | 0.90 |
| `nlp-large` | 493 | 121,278 | 0.80 | 51 | 3,000 | 0.90 |
| `nlp-small` | 200 | 19,900 | 0.90 | 0 | 500 | 1.0 |
| `edge-cases` | 20 | 190 | 0.90 | 22 | 190 | 1.0 |

The NLP keywords are research concepts embedded with `all-MiniLM-L6-v2`, projected to 32 dimensions
with a seeded Gaussian matrix, normalized, scaled by 10⁶ and rounded to integers. `edge-cases` holds
hand-built vectors: duplicate directions, orthogonal and opposite vectors, and pairs whose cosine is
exactly 0.70, 0.80 or 0.90. The generator lives outside this repository; only its JSON output is here.

## Layout

It has the standard plugin layout, which NanoAgent requires:

```text
.
├── nanoagent.yaml   # manifest (a slimconfig ManifestConfig): id, runtime, permissions, tools, tasks
├── contract.py      # each tool's and task's typed input and output: slimconfig Output classes
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .env             # needed environment variable names (none here)
├── src/plugin/      # the MCP server (__main__.py, its IMPLEMENTATIONS table), one module per tool,
│                    # bundle.py (datasets and paths), rule.py (the exact rule) and data/ (the JSON)
├── tests/           # test_tools.py and test_verifier.py
├── ui/              # renderers.js and renderers.css: outputs are drawn from their JSON Schema
├── product/         # the product page a catalog shows: product.yaml and icon.svg
└── verifier/        # deterministic verifier package, one module per tool, find.py for the task
```

`contract.py` types every value: the server builds each call's input into its class and checks its
output against its own, and NanoAgent and the verifier get the same classes. A failing tool raises,
and the MCP server returns an error result.

`verifier/` is deterministic: it imports only the standard library and `contract`, and reads only
the workspace's `candidates/` and `score_ledger.json`. Beyond the task check, it rederives every tool
call's meaning: the run id, the ranking and histogram against the saved candidates, and for each
`score_pairs` call that the ledger scored exactly the range's new pairs, that the qualifying pairs
meet the rule by their own integers, and that the totals and remaining budget add up.

## Develop and run

```bash
uv lock
uv sync --extra dev
uv run pytest
uv run ruff check .
uv lock --check
```

NanoAgent passes the folder the plugin may write to as `NANOAGENT_WORKSPACE`; `lsh_candidates`
writes `candidates/<run_id>.json` there and `score_pairs` writes `score_ledger.json`.
