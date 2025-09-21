"""Deepeval-based evaluation service for summary classification.

This module provides comprehensive evaluation using Deepeval framework:
- Classification accuracy metrics
- Contextual precision/recall/relevancy
- Faithfulness evaluation
- Live evaluation capabilities
"""
from typing import Dict, Any, List, Iterable, Tuple, Optional
import os

# Deepeval imports with fallback
try:
    from deepeval import evaluate
    from deepeval.metrics import (
        AccuracyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
        GEval
    )
    from deepeval.test_case import LLMTestCase
    from deepeval.dataset import EvaluationDataset
    from deepeval.models import DeepEvalBaseLLM
    import google.generativeai as genai
    HAS_DEEPEVAL = True
except ImportError:
    HAS_DEEPEVAL = False
    # Create dummy classes for when Deepeval is not available
    class DeepEvalBaseLLM:
        pass
    class LLMTestCase:
        def __init__(self, **kwargs):
            pass
    class AccuracyMetric:
        pass
    class ContextualPrecisionMetric:
        def __init__(self, **kwargs):
            pass
    class ContextualRecallMetric:
        def __init__(self, **kwargs):
            pass
    class ContextualRelevancyMetric:
        def __init__(self, **kwargs):
            pass
    class FaithfulnessMetric:
        def __init__(self, **kwargs):
            pass
    def evaluate(*args, **kwargs):
        return {}
    # Create dummy genai module
    class DummyGenai:
        @staticmethod
        def configure(**kwargs):
            pass
        class GenerativeModel:
            def __init__(self, *args, **kwargs):
                pass
            def generate_content(self, *args, **kwargs):
                class Response:
                    text = "Dummy response"
                return Response()
    genai = DummyGenai()
    print("Warning: Deepeval not available. Install with: pip install deepeval")


def _safe_iter(x):
    """Return an iterable of records whether x is a single dict or a list."""
    if x is None:
        return []
    if isinstance(x, dict):
        return [x]
    return list(x)


class GeminiLLM(DeepEvalBaseLLM):
    """Custom Gemini LLM wrapper for Deepeval."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name
        if HAS_DEEPEVAL:
            # Configure Gemini
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.model = genai.GenerativeModel(model_name)
        else:
            # Use dummy model when Deepeval is not available
            self.model = None
    
    def load_model(self):
        return self.model
    
    def generate(self, prompt: str) -> str:
        if not HAS_DEEPEVAL or self.model is None:
            return "Dummy response - Deepeval not available"
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini generation error: {e}")
            return "Error generating response"
    
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
    
    def get_model_name(self) -> str:
        return self.model_name


class EvaluationService:
    def __init__(self, use_deepeval: bool = True):
        self.use_deepeval = use_deepeval and HAS_DEEPEVAL
        if self.use_deepeval:
            # Initialize Gemini LLM for Deepeval
            self.gemini_llm = GeminiLLM()
            print("✓ Using Deepeval framework with Gemini for evaluation")
        else:
            # Initialize dummy LLM for fallback
            self.gemini_llm = None
            print("⚠ Using fallback evaluation (install deepeval for advanced metrics)")

    def evaluate(self, predicted: Iterable[Dict[str, Any]], ground_truth: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate predictions using Deepeval or fallback metrics.

        Args:
            predicted: iterable of dicts containing at least 'doc_type'
            ground_truth: iterable of dicts containing at least 'doc_type'

        Returns:
            A dict with comprehensive evaluation metrics
        """
        if self.use_deepeval:
            return self._evaluate_with_deepeval(predicted, ground_truth)
        else:
            return self._evaluate_fallback(predicted, ground_truth)

    def _evaluate_with_deepeval(self, predicted: Iterable[Dict[str, Any]], ground_truth: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate using Deepeval framework."""
        preds = _safe_iter(predicted)
        gts = _safe_iter(ground_truth)

        if not preds or not gts:
            return {"error": "No predictions or ground truth provided"}

        # Create test cases for Deepeval
        test_cases = []
        for i, (pred, gt) in enumerate(zip(preds, gts)):
            predicted_doc_type = pred.get("doc_type", "UNKNOWN")
            actual_doc_type = gt.get("doc_type", "UNKNOWN")
            
            # Create context from summary if available
            context = gt.get("summary", "")
            
            test_case = LLMTestCase(
                input=context,
                actual_output=predicted_doc_type,
                expected_output=actual_doc_type,
                context=context
            )
            test_cases.append(test_case)

        # Define metrics with Gemini LLM
        if self.gemini_llm is not None:
            metrics = [
                AccuracyMetric(),
                ContextualPrecisionMetric(model=self.gemini_llm),
                ContextualRecallMetric(model=self.gemini_llm),
                ContextualRelevancyMetric(model=self.gemini_llm),
                FaithfulnessMetric(model=self.gemini_llm)
            ]
        else:
            metrics = [AccuracyMetric()]

        # Run evaluation
        try:
            results = evaluate(test_cases, metrics)
            
            # Extract metrics
            accuracy = results.get("accuracy", 0.0)
            contextual_precision = results.get("contextual_precision", 0.0)
            contextual_recall = results.get("contextual_recall", 0.0)
            contextual_relevancy = results.get("contextual_relevancy", 0.0)
            faithfulness = results.get("faithfulness", 0.0)

            return {
                "framework": "deepeval",
                "n": len(test_cases),
                "accuracy": accuracy,
                "contextual_precision": contextual_precision,
                "contextual_recall": contextual_recall,
                "contextual_relevancy": contextual_relevancy,
                "faithfulness": faithfulness,
                "overall_score": (accuracy + contextual_precision + contextual_recall + contextual_relevancy + faithfulness) / 5
            }
        except Exception as e:
            print(f"Deepeval evaluation failed: {e}")
            return self._evaluate_fallback(predicted, ground_truth)

    def _evaluate_fallback(self, predicted: Iterable[Dict[str, Any]], ground_truth: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback evaluation using basic metrics."""
        preds = _safe_iter(predicted)
        gts = _safe_iter(ground_truth)

        # If lengths mismatch, evaluate up to the shorter one
        n = min(len(preds), len(gts))
        if n == 0:
            return {"framework": "fallback", "accuracy": 0.0, "details": {}, "n": 0}

        preds = preds[:n]
        gts = gts[:n]

        # collect labels
        labels = set()
        for p, g in zip(preds, gts):
            labels.add((p.get("doc_type") or "").upper())
            labels.add((g.get("doc_type") or "").upper())

        labels.discard("")

        # initialize counts
        tp = {lab: 0 for lab in labels}
        fp = {lab: 0 for lab in labels}
        fn = {lab: 0 for lab in labels}
        correct = 0

        for p, g in zip(preds, gts):
            p_lab = (p.get("doc_type") or "").upper()
            g_lab = (g.get("doc_type") or "").upper()
            if p_lab == g_lab and p_lab != "":
                correct += 1
                if p_lab in tp:
                    tp[p_lab] += 1
            else:
                # false positive for predicted label (if non-empty)
                if p_lab and p_lab in fp:
                    fp[p_lab] += 1
                # false negative for ground-truth label (if non-empty)
                if g_lab and g_lab in fn:
                    fn[g_lab] += 1

        # per-label metrics
        per_label: Dict[str, Dict[str, float]] = {}
        sum_tp = sum(tp.values())
        sum_fp = sum(fp.values())
        sum_fn = sum(fn.values())

        for lab in sorted(labels):
            t = tp.get(lab, 0)
            f_p = fp.get(lab, 0)
            f_n = fn.get(lab, 0)
            precision = t / (t + f_p) if (t + f_p) > 0 else 0.0
            recall = t / (t + f_n) if (t + f_n) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            per_label[lab] = {"precision": precision, "recall": recall, "f1": f1, "support": t + f_n}

        # micro averages use global tp/fp/fn
        micro_precision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
        micro_recall = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

        # macro averages: simple average of per-label metrics
        if per_label:
            macro_precision = sum(v["precision"] for v in per_label.values()) / len(per_label)
            macro_recall = sum(v["recall"] for v in per_label.values()) / len(per_label)
            macro_f1 = sum(v["f1"] for v in per_label.values()) / len(per_label)
        else:
            macro_precision = macro_recall = macro_f1 = 0.0

        accuracy = correct / n

        return {
            "framework": "fallback",
            "n": n,
            "accuracy": accuracy,
            "per_label": per_label,
            "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1},
            "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
        }

    def evaluate_single(self, predicted_doc_type: str, actual_doc_type: str, context: str = "") -> Dict[str, Any]:
        """Evaluate a single prediction using Deepeval."""
        if not self.use_deepeval:
            return {"framework": "fallback", "accuracy": 1.0 if predicted_doc_type == actual_doc_type else 0.0}

        try:
            test_case = LLMTestCase(
                input=context,
                actual_output=predicted_doc_type,
                expected_output=actual_doc_type,
                context=context
            )

            if self.gemini_llm is not None:
                metrics = [
                    AccuracyMetric(),
                    ContextualPrecisionMetric(model=self.gemini_llm),
                    ContextualRecallMetric(model=self.gemini_llm),
                    ContextualRelevancyMetric(model=self.gemini_llm),
                    FaithfulnessMetric(model=self.gemini_llm)
                ]
            else:
                metrics = [AccuracyMetric()]

            results = evaluate([test_case], metrics)
            
            return {
                "framework": "deepeval",
                "accuracy": results.get("accuracy", 0.0),
                "contextual_precision": results.get("contextual_precision", 0.0),
                "contextual_recall": results.get("contextual_recall", 0.0),
                "contextual_relevancy": results.get("contextual_relevancy", 0.0),
                "faithfulness": results.get("faithfulness", 0.0),
            }
        except Exception as e:
            print(f"Single evaluation failed: {e}")
            return {"framework": "fallback", "accuracy": 1.0 if predicted_doc_type == actual_doc_type else 0.0}
