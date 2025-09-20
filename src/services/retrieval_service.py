"""Retrieval service that queries the vector DB using triplet keys."""
from typing import List, Tuple, Dict, Any


class RetrievalService:
    def __init__(self, vector_db):
        self.vector_db = vector_db

    def retrieve_by_triplets(self, triplets: List[Tuple[str, str, str]]) -> Dict[str, Any]:
        keys = [" | ".join(t) for t in triplets]
        record = self.vector_db.query(keys)
        return record or {"doc_type": None, "doc_code": None}
