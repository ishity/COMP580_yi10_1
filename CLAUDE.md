# CLAUDE.md: build the `lsh-similar-pairs` NanoAgent plugin

## Context

This repository is a NanoAgent plugin for the COMP 480/580 milestone "Verifiable Tasks" (due 2026-11-02). It was cloned from the reference plugin `zeyuyang8/arxiv-source`, its git history was removed, and a fresh repo was initialized on branch `main`. Nothing else has been changed yet.

Goal: replace every arXiv-specific part (tools, tasks, verifier, tests, UI, product page, README) with an original task: **finding all semantically similar keyword pairs with SimHash LSH under a scoring budget**, checked by a deterministic verifier against an exact brute-force reference.

This is plugin 1 of 5. The data format and verifier helpers you write here will be reused in the next four plugins, so keep them clean and general.

## Read first

1. Clone the NanoAgent repo next to this one (not inside it) and read the official plugin guide in full:
   ```bash
   git clone --depth 1 -b production https://github.com/RUSH-LAB/nanoagent.git ../nanoagent
   ```
   - `../nanoagent/docs/instructions/skills/publish-plugin/SKILL.md` (the authoritative rules)
   - `../nanoagent/AGENTS.md`
   - `../nanoagent/nanoagent/test/plugins/word-check/` (a smaller example plugin)
2. Read every file of this repository (the arxiv-source reference) before changing anything. When unsure how a piece should look, copy the pattern from it. Do not guess.

If anything in this file conflicts with SKILL.md, SKILL.md wins. Tell me about the conflict.

## Hard rules

- **Package management is uv only.** No pip, conda or pixi. Keep the template's `requires-python`, its dependencies and its `slimconfig` source unchanged. New dependencies need lower and upper bounds. Run `uv lock` after any `pyproject.toml` change.
- **The verifier is deterministic code.** No LLM anywhere in `verifier/`. Pure standard library only (`json`, `math`, `fractions`, `pathlib`, `dataclasses`, `enum`, `collections`, `itertools`, ...) plus `contract` and its own package. No clock, randomness, environment variables, network, subprocesses, `open`, `eval`, `exec`, `getattr`, private attributes or file writes. Read workspace files with `Path(...).read_text()`. Only `.py` files in `verifier/`. It runs twice and both verdicts must match.
- **Rules go in code, judgment goes to the model.** Anything computable by rule is computed by a tool. The model only chooses parameters and which tool to call next.
- **No fallbacks.** A tool that cannot do its job raises. It never returns a partial or made-up value.
- **Top level holds only** `nanoagent.yaml`, `contract.py`, `README.md`, `LICENSE`, `pyproject.toml`, `uv.lock`, `.gitignore`, `.env`, `src/`, `tests/`, `verifier/`, `ui/`, `product/`, and agent instruction files. No reports, logs, notes or scripts at the top level.
- **Ask me before** creating a GitHub repository, making anything public, or pushing. I submit to RUSH Lab myself.
- Do not use em-dashes in any text you write (README, product page, docstrings, comments).

## Task design

### The task in one sentence

Given a bundled keyword dataset with embeddings and a similarity threshold τ, find every keyword pair whose cosine similarity is at least τ, using SimHash LSH to choose which pairs to score, without exceeding a budget on the number of scored pairs.

### Why it is objectively verifiable

The correct answer is a set fully determined by the bundled data and τ. Similarity is decided in **exact integer and Fraction arithmetic** (see below), so there is no floating-point ambiguity at the threshold. Correctness of every submitted pair, completeness (recall) and budget compliance are all computable by code.

### Data (bundled, offline)

- Datasets live in `src/plugin/data/<dataset>.json`. Each file holds:
  - `keywords`: list of unique keyword strings (NLP research concepts, e.g. "machine translation", "long-context modeling").
  - `vectors`: one integer vector per keyword. These are L2-normalized embeddings, projected to 32 dimensions, scaled by 10^6 and rounded to integers.
- `src/plugin/data/hyperplanes.json`: 256 fixed integer hyperplanes of dimension 32, generated once with a fixed seed.
- Datasets (as an Enum `Dataset` in the contract):
  - `nlp-small`: about 200 keywords.
  - `nlp-large`: about 500 keywords.
  - `edge-cases`: about 20 hand-built vectors covering duplicates in direction, orthogonal vectors, opposite vectors, and pairs whose cosine equals a supported τ exactly.
- **The data generator does not live in this repo.** Put it in a sibling folder `../lsh-data-gen/` with its own uv project. It may use `sentence-transformers` (e.g. `all-MiniLM-L6-v2`) and numpy, applies a seeded random projection to 32 dims, normalizes, scales and rounds, then writes the JSON files into this repo. Record the exact command, model name and seed in `../lsh-data-gen/README.md`. Commit only the generated JSON here.

### Exact similarity rule (shared by tools and verifier)

For integer vectors `u`, `v` and threshold τ given as a decimal string (e.g. `"0.80"`, parsed with `Fraction`):

- `dot = Σ u_i v_i`, `nu = Σ u_i²`, `nv = Σ v_i²` (all integers).
- The pair qualifies iff `dot >= 0` and `dot² >= τ² · nu · nv`, compared as Fractions.

Implement this once per side (tool side in `src/plugin/`, verifier side in `verifier/`) as small, identical pure functions, and test that they agree.

### Tools (one module per tool in `src/plugin/`)

| Tool | Input | Output | Notes |
|---|---|---|---|
| `describe_dataset` | `dataset` | keyword count, dimension, number of hyperplanes available, total pair count | |
| `lsh_candidates` | `dataset`, `bits_per_band` (int), `bands` (int) | candidate pairs (keyword pairs, canonical order a < b, no duplicates), their count | Uses the first `bits_per_band × bands` hyperplanes; raises if that exceeds 256 or any value is < 1. Band signature = sign bits of the integer dot products (treat 0 as positive). Two keywords are candidates if any band signature matches. |
| `score_pairs` | `dataset`, `pairs`, `threshold` | the subset of pairs that qualify under the exact rule, plus how many pairs this call scored and the cumulative total | Appends every scored pair to `workspace/score_ledger.json`. Raises, without scoring anything, if the call would push the cumulative total past the budget. Raises on unknown keywords, self-pairs, or duplicates within the call. |
| `exact_pairs` | `dataset`, `threshold` | all qualifying pairs by brute force | **Reference tool only.** Not listed in the task's `tools`, so the model never sees it. |

The budget must be known to `score_pairs`. Pass it through the task input and have the agent supply it, or store it in the ledger at the first call: pick the design that SKILL.md's rules allow, and explain the choice to me before implementing.

### Task

- id `find-similar-pairs`, runner `agent`, tools `[describe_dataset, lsh_candidates, score_pairs]`, reference `exact_pairs`.
- Input: `dataset`, `threshold` (decimal string), `budget` (max total scored pairs), `min_recall` (decimal string, e.g. `"0.90"`).
- Output: the list of qualifying pairs.
- Budgets must make brute force impossible: for `nlp-large` (about 125,000 pairs) use a budget of a few thousand.
- Instructions to the model: describe the dataset, choose LSH parameters, generate candidates, score them within budget, optionally rerun LSH with different parameters to raise recall if budget remains, then submit every qualifying pair returned by `score_pairs`.

### Verifier (`verify_task`)

Checks, in this order. The first failure decides the verdict.

1. **Malformed**: every pair names two distinct known keywords; no duplicate pairs (treat (a, b) and (b, a) as the same).
2. **Budget**: read `workspace/score_ledger.json`. Missing ledger with a non-empty answer is a failure. Total scored pairs must be at most `budget`.
3. **Precision**: every submitted pair is in the reference set. Any false pair makes the result invalid with score 0.
4. **Provenance**: every submitted pair appears as qualifying in the ledger, so the agent cannot copy pairs it never scored.
5. **Recall**: `recall = |submitted| / |reference|`. If the reference is empty, recall is 1 when the submission is also empty. `valid` iff recall ≥ `min_recall`. `score = recall`.

`verify_input` and `verify_output` check each tool call's meaning (canonical ordering, counts consistent with lists, returned qualifying pairs are a subset of the input pairs, cumulative totals non-decreasing), not shape.

### Tests

- `tests/test_verifier.py`: correct answer; one wrong pair; empty answer with non-empty reference; empty reference; duplicate and reversed pairs; unknown keyword; self-pair; budget exceeded; missing ledger; pair submitted but never scored; recall exactly at `min_recall`; a pair whose cosine equals τ exactly. Build values the way the sandbox does, with `build_output(plain_output(...), Class)`.
- Tool tests: determinism of `lsh_candidates`, exact-rule agreement between tool and verifier implementations on all `edge-cases` pairs, budget enforcement in `score_pairs`, and that `exact_pairs` matches a slow reference implementation on `nlp-small`.

## Work plan and checkpoints

Stop at each checkpoint, summarize what you did, and wait for my OK.

1. Read the files listed in "Read first". Then rename the plugin everywhere (`nanoagent.yaml` id and name, README title, `pyproject.toml` name `nanoagent-lsh-similar-pairs`, LICENSE holder "Yuri" and year 2026). Set `status: in-development`. Rename branch to `main` if it is not already. **Checkpoint 1**: show me the diff and any SKILL.md rule you think affects the design above.
2. Build `../lsh-data-gen/` and generate the data. **Checkpoint 2**: show dataset sizes and, for τ in {0.70, 0.80, 0.90}, the number of qualifying pairs per dataset, so we can pick sensible τ and budget values.
3. Write `contract.py`. **Checkpoint 3**: show it to me before writing tools.
4. Implement tools, manifest, verifier and tests.
5. Write `ui/` (start with `export default {};` and an empty CSS file), `product/product.yaml` and `product/icon.svg`, and the README sections `## Tools`, `## Layout`, `## Develop and run` in that order.
6. Run all checks until they pass:
   ```bash
   uv lock
   uv sync --extra dev
   uv run pytest
   uv run ruff check .
   uv lock --check
   ```
   Then walk the manual checklist in SKILL.md section 8. Set `status: ready`. **Checkpoint 4**: report results.
7. With my OK, create the public GitHub repo and push.

## Notes for the report (keep in `../lsh-report-notes.md`, never in this repo)

As you go, record: the data generation command and seed, the chosen τ and budgets with reasons, every verifier case and what it returns, test output, and known limitations (for example: embeddings come from one model; LSH recall depends on the agent's parameter choice; efficiency in time and memory is not measured by the verifier because it cannot read the clock).