"""Minimal stub for a Gemini-like LLM client used for triplet extraction.

This is intentionally simplistic: it performs heuristic extraction so the POC
is runnable without external LLM calls.
"""
import re
from typing import List, Tuple


class GeminiClient:
    def extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """Heuristic triplet extractor.

        Returns a list of (subject, predicate, object) tuples.
        """
        triplets = []
        # Very small heuristics: look for currency amounts, dates, invoice/order numbers
        amount_match = re.search(r"\$\s?\d+[\d,]*", text)
        if amount_match:
            triplets.append(("AMOUNT", "has_value", amount_match.group(0)))
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", text)
        if date_match:
            triplets.append(("DATE", "on", date_match.group(0)))
        # invoice/order identifiers like INV-100 or PO-2025-01
        id_match = re.search(r"(INV|PO|BS|ACC)[-_]?[A-Z0-9]+", text, re.IGNORECASE)
        if id_match:
            triplets.append(("ID", "identifier", id_match.group(0)))
        # organization names (naive: capitalized words sequence)
        org_match = re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text)
        if org_match:
            triplets.append(("ORG", "name", org_match.group(0)))
        # fallback: split into subject-verb-object using 'was' or 'show'
        m = re.search(r"([A-Za-z ]+) (was|shows|issued|submitted|made) (.+)\.", text)
        if m:
            triplets.append((m.group(1).strip(), m.group(2).strip(), m.group(3).strip().rstrip('.')))
        return triplets
