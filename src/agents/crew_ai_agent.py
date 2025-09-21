"""CrewAI orchestrator agent scaffold.

This is a minimal, local-only implementation to mirror the README POC flow.
It uses simple, deterministic components to extract triplets, normalize/mask
PII, store/query an in-memory vector DB, and evaluate the result.
"""
from typing import Dict, Any, List
from pathlib import Path
import json

from services.triplet_service import TripletService
from db.vector_store import FaissVectorStore
from services.retrieval_service import RetrievalService
from services.evaluation_service import EvaluationService


class CrewAIAgent:
    def __init__(self, data_path: str = None):
        self.data_path = data_path or str(Path(__file__).resolve().parents[2] / "data" / "summaries.json")
        # Initialize components
        self.triplet_service = TripletService()
        self.vector_db = FaissVectorStore(model_name="all-MiniLM-L6-v2")
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
        
        # Collect all triplets and metadata for batch processing
        all_triplets = []
        all_metas = []
        
        for item in dataset:
            summary = item.get("summary", "")
            doc_type = item.get("doc_type")
            doc_code = item.get("doc_code")
            triplets = self.triplet_service.extract_and_normalize(summary)
            
            # Convert triplets to text format for embedding
            for t in triplets:
                triplet_text = " | ".join(t)
                all_triplets.append(triplet_text)
                all_metas.append({
                    "doc_type": doc_type, 
                    "doc_code": doc_code, 
                    "summary": summary,
                    "triplet": triplet_text
                })
        
        # Batch add all triplets to vector store
        if all_triplets:
            self.vector_db.add(all_triplets, all_metas)
            print(f"Indexed {len(all_triplets)} triplets from {len(dataset)} summaries")

    def run(self, summary: str, ground_truth_doc_type: str = None) -> Dict[str, Any]:
        """Run the POC pipeline on a single summary string.

        Args:
            summary: The document summary to classify
            ground_truth_doc_type: Optional ground truth for evaluation

        Returns a dict with keys: summary_type, doc_code, metrics
        """
        # Extract triplets from summary
        triplets = self.triplet_service.extract_and_normalize(summary)
        
        # Retrieve document type and code
        predicted = self.retrieval.retrieve_by_triplets(triplets)
        predicted_doc_type = predicted.get("doc_type") if predicted else None
        predicted_doc_code = predicted.get("doc_code") if predicted else None
        
        # Live evaluation if ground truth is provided
        if ground_truth_doc_type:
            metrics = self.evaluator.evaluate_single(
                predicted_doc_type=predicted_doc_type or "UNKNOWN",
                actual_doc_type=ground_truth_doc_type,
                context=summary
            )
        else:
            # Basic metrics without ground truth
            metrics = {
                "framework": "no_ground_truth",
                "predicted_doc_type": predicted_doc_type,
                "confidence": "N/A",
                "triplets_extracted": len(triplets)
            }
        
        return {
            "summary_type": predicted_doc_type,
            "doc_code": predicted_doc_code,
            "metrics": metrics,
            "triplets": triplets,
            "summary": summary
        }

    def run_batch_evaluation(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run batch evaluation on multiple test cases.
        
        Args:
            test_cases: List of dicts with 'summary' and 'doc_type' keys
            
        Returns:
            Comprehensive evaluation results
        """
        predictions = []
        ground_truths = []
        
        for test_case in test_cases:
            summary = test_case.get('summary', '')
            expected_doc_type = test_case.get('doc_type', '')
            
            # Run prediction
            result = self.run(summary, expected_doc_type)
            
            predictions.append({
                "doc_type": result["summary_type"],
                "doc_code": result["doc_code"],
                "summary": summary
            })
            
            ground_truths.append({
                "doc_type": expected_doc_type,
                "summary": summary
            })
        
        # Run comprehensive evaluation
        evaluation_results = self.evaluator.evaluate(predictions, ground_truths)
        
        return {
            "batch_evaluation": evaluation_results,
            "individual_results": [
                {
                    "summary": pred["summary"],
                    "predicted": pred["doc_type"],
                    "actual": gt["doc_type"],
                    "correct": pred["doc_type"] == gt["doc_type"]
                }
                for pred, gt in zip(predictions, ground_truths)
            ]
        }
