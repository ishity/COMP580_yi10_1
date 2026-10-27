"""The exact similarity rule, in integer and Fraction arithmetic only.

src/plugin/rule.py and verifier/rule.py hold this same file, so the tools and the verifier decide alike.
"""

import re
from fractions import Fraction

DECIMAL = re.compile(r"^[0-9]+(?:\.[0-9]{1,3})?$")
DECIMAL_RULE = 'a decimal string in (0, 1] with at most 3 decimal places, e.g. "0.80"'


def parse_decimal(text):
    """``text`` as a Fraction, or None when it is not DECIMAL_RULE."""
    if not DECIMAL.fullmatch(text):
        return None
    value = Fraction(text)
    return value if 0 < value <= 1 else None


def qualifies(dot, norm_a, norm_b, threshold):
    """Whether a pair's cosine, dot / sqrt(norm_a * norm_b), is at least ``threshold`` (a Fraction)."""
    return dot >= 0 and dot * dot >= threshold * threshold * norm_a * norm_b
