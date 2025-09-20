"""A tiny in-memory vector DB placeholder for the POC.

It stores string keys (triplet text) and associated metadata. Query is a
simple substring match.
"""
from typing import Dict, List, Any


class VectorDBClient:
    def __init__(self):
        # key -> list of records
        self.store_map: Dict[str, List[Dict[str, Any]]] = {}

    def store(self, key: str, record: Dict[str, Any]):
        self.store_map.setdefault(key, []).append(record)

    def query(self, key_candidates: List[str]):
        """Return the first matching record for any candidate key, or None."""
        for k in key_candidates:
            for stored_key, records in self.store_map.items():
                if k in stored_key or stored_key in k:
                    # return the most recent
                    return records[-1]
        return None
