"""Retrieval service that queries the vector DB using triplet keys."""
from typing import List, Tuple, Dict, Any
from utils.normalization import triplet_to_text


class RetrievalService:
    def __init__(self, vector_db):
        self.vector_db = vector_db

    def index_triplets(self, triplets: List[Tuple[str, str, str]], metas: List[Dict[str, Any]]):
        """Index a batch of triplet texts with associated metadata.

        metas must be aligned with triplets and contain at least 'doc_type' and 'doc_code'.
        """
        texts = [triplet_to_text(t) for t in triplets]
        self.vector_db.add(texts, metas)

    def retrieve_by_triplet(self, triplet: Tuple[str, str, str], top_k: int = 5) -> Dict[str, Any]:
        query_text = triplet_to_text(triplet)
        # Try the modern vector_db.query(query_text, top_k=...) signature first.
        try:
            hits = self.vector_db.query(query_text, top_k=top_k)
        except TypeError:
            # Fallback to older simple VectorDBClient that expects a list of key candidates
            keys = [query_text]
            rec = self.vector_db.query(keys)
            hits = [rec] if rec else []
        if not hits:
            return {"doc_type": None, "doc_code": None, "hits": []}

        # aggregate by doc_type, sum scores
        agg: Dict[str, Dict[str, Any]] = {}
        for h in hits:
            dt = h.get("doc_type") or "UNKNOWN"
            dc = h.get("doc_code") or ""
            score = h.get("score", 0.0)
            if dt not in agg:
                agg[dt] = {"score": 0.0, "doc_codes": {dc: score}}
            agg[dt]["score"] += score
            agg[dt]["doc_codes"][dc] = max(agg[dt]["doc_codes"].get(dc, 0.0), score)

        # pick best doc_type
        best = max(agg.items(), key=lambda kv: kv[1]["score"])  # (doc_type, data)
        doc_type = best[0]
        doc_code = max(best[1]["doc_codes"].items(), key=lambda kv: kv[1])[0]
        return {"doc_type": doc_type, "doc_code": doc_code, "hits": hits}

    def retrieve_by_triplets(self, triplets: List[Tuple[str, str, str]]) -> Dict[str, Any]:
        """Backward-compatible: accept a list of triplets and aggregate retrievals.

        Strategy: call `retrieve_by_triplet` for each triplet and vote on the best
        doc_type/doc_code. This keeps the previous CrewAIAgent behavior while
        supporting the richer Faiss-backed retrieval.
        """
        votes: Dict[str, Dict[str, float]] = {}
        for t in triplets:
            res = self.retrieve_by_triplet(t, top_k=3)
            dt = res.get("doc_type")
            dc = res.get("doc_code")
            # if retrieval returned no doc_type, skip
            if not dt:
                continue
            votes.setdefault(dt, {}).setdefault(dc, 0.0)
            # aggregate by sum of top hit scores if available
            if res.get("hits"):
                votes[dt][dc] += sum(h.get("score", 0.0) for h in res.get("hits"))
            else:
                votes[dt][dc] += 1.0

        if not votes:
            return {"doc_type": None, "doc_code": None, "hits": []}

        # pick best doc_type/doc_code by summed score
        best_dt, dc_map = max(votes.items(), key=lambda kv: sum(kv[1].values()))
        best_dc = max(dc_map.items(), key=lambda kv: kv[1])[0]
        return {"doc_type": best_dt, "doc_code": best_dc, "hits": []}
