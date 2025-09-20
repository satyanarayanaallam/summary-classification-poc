"""Normalization and PII masking utilities."""
import re
from typing import Tuple, List


def mask_pii(text: str) -> str:
    """Mask numeric sequences (accounts, ids) and credit-card-like groups."""
    # Replace long digit sequences with <NUM>
    masked = re.sub(r"\b\d{4,}\b", "<NUM>", text)
    # Replace shorter numbers (like amounts) with <NUM>
    masked = re.sub(r"\b\d+\b", "<NUM>", masked)
    return masked


def normalize_triplet(triplet: Tuple[str, str, str]) -> Tuple[str, str, str]:
    s, p, o = triplet
    return (mask_pii(s).upper(), p.lower(), mask_pii(o).upper())
