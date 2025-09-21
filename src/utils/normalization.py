"""Normalization utilities for triplet extraction POC.

This module provides lightweight canonicalization for triplets:
- lowercasing
- predicate normalization (common predicate mapping)
- numeric token replacement for amounts/dates/account numbers
- conversion to a canonical triplet text suitable for embedding

Note: per POC constraints we do not perform PII masking here; assume
input summaries are already free of personal data or that masking is
handled upstream if required.
"""
import re
from typing import Dict, Tuple


PREDICATE_MAP: Dict[str, str] = {
    "has_value": "has_amount",
    "amount": "has_amount",
    "hasamount": "has_amount",
    "issued": "issued_by",
    "issued_by": "issued_by",
    "issued_to": "issued_to",
    "name": "issued_by",
    "identifier": "identifier",
    "on": "date",
}


AMOUNT_RE = re.compile(r"\$\s?[0-9,]+(?:\.[0-9]{2})?")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
ACC_RE = re.compile(r"\b\d{6,}\b")


def _replace_tokens(text: str) -> str:
    """Replace numeric tokens with canonical placeholders."""
    t = AMOUNT_RE.sub("<AMOUNT>", text)
    t = DATE_RE.sub("<DATE>", t)
    t = ACC_RE.sub("<ACCOUNT_NO>", t)
    return t


def normalize_triplet(triplet: Tuple[str, str, str]) -> Tuple[str, str, str]:
    """Normalize a (subject, predicate, object) triplet.

    - Lowercase subject and object
    - Map predicate to canonical predicate
    - Replace numeric tokens in object with placeholders
    """
    subj, pred, obj = triplet
    subj_n = subj.strip().lower()
    pred_n = pred.strip().lower().replace(" ", "_")
    pred_mapped = PREDICATE_MAP.get(pred_n, pred_n)
    obj_n = _replace_tokens(obj.strip().lower())
    return subj_n, pred_mapped, obj_n


def triplet_to_text(triplet: Tuple[str, str, str]) -> str:
    """Convert a normalized triplet to a canonical single-line text for embedding.

    Example: ("invoice", "has_amount", "<AMOUNT>") -> "invoice has_amount <AMOUNT>"
    """
    s, p, o = normalize_triplet(triplet)
    # remove any extra whitespace
    parts = [s, p, o]
    return " ".join([p for p in parts if p])
