"""Where score_pairs keeps a task instance's ledger: one file per dataset, threshold and budget.

NanoAgent copies a member's workspace files into every run and back, so ledgers outlive a run. Naming
each ledger by the values it locks keeps instances apart: two instances never share a ledger, and a
rerun of one instance finds its own. src/plugin/ledgers.py and verifier/ledgers.py hold this same file.
"""

from fractions import Fraction
from pathlib import Path

LEDGERS = "ledgers"


def threshold_text(threshold):
    """``threshold`` (a Fraction with at most 3 decimal places) written by value to 3 places: 4/5 is 0.800."""
    thousandths = threshold * 1000
    if thousandths.denominator != 1:
        raise ValueError(f"threshold {threshold} has more than 3 decimal places")
    whole, rest = divmod(thousandths.numerator, 1000)
    return f"{whole}.{rest:03d}"


def ledger_name(dataset, threshold, budget):
    """The ledger's file name for a dataset (its name), threshold (a Fraction) and budget (an int)."""
    return f"{dataset}-{threshold_text(Fraction(threshold))}-{budget}.json"


def ledger_path(workspace, dataset, threshold, budget):
    return Path(workspace) / LEDGERS / ledger_name(dataset, threshold, budget)
