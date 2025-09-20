"""A tiny evaluation stub that mimics Deepeval's metrics."""
from typing import Dict, Any


class EvaluationService:
    def evaluate(self, predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        # Since we may not have ground truth in run(), return placeholder metrics
        return {"accuracy": 1.0 if predicted and predicted.get("doc_type") else 0.0}
