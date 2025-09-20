"""CrewAI orchestrator agent scaffold.

This is a minimal, local-only implementation to mirror the README POC flow.
It uses simple, deterministic components to extract triplets, normalize/mask
PII, store/query an in-memory vector DB, and evaluate the result.
"""
from typing import Dict, Any
from pathlib import Path
import json

from src.services.triplet_service import TripletService
from src.db.vector_db import VectorDBClient
from src.services.retrieval_service import RetrievalService
from src.services.evaluation_service import EvaluationService


class CrewAIAgent:
    def __init__(self, data_path: str = None):
        self.data_path = data_path or str(Path(__file__).resolve().parents[2] / "data" / "summaries.json")
        # Initialize components
        self.triplet_service = TripletService()
        self.vector_db = VectorDBClient()
        self.retrieval = RetrievalService(self.vector_db)
        self.evaluator = EvaluationService()
        # Load and index sample data
        self._load_and_index()

    def _load_and_index(self):
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
        except FileNotFoundError:
            dataset = []
        for item in dataset:
            summary = item.get("summary", "")
            doc_type = item.get("doc_type")
            doc_code = item.get("doc_code")
            triplets = self.triplet_service.extract_and_normalize(summary)
            # For simplicity store a text key per triplet in the vector DB
            for t in triplets:
                key = " | ".join(t)
                self.vector_db.store(key, {"doc_type": doc_type, "doc_code": doc_code, "summary": summary})

    def run(self, summary: str) -> Dict[str, Any]:
        """Run the POC pipeline on a single summary string.

        Returns a dict with keys: summary_type, doc_code, metrics
        """
        triplets = self.triplet_service.extract_and_normalize(summary)
        predicted = self.retrieval.retrieve_by_triplets(triplets)
        # Evaluate against a best-effort ground truth from dataset (if any)
        metrics = self.evaluator.evaluate(predicted, None)
        return {
            "summary_type": predicted.get("doc_type") if predicted else None,
            "doc_code": predicted.get("doc_code") if predicted else None,
            "metrics": metrics,
        }
