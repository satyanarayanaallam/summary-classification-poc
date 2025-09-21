"""Deepeval test cases for summary classification evaluation.

This module creates comprehensive test cases for evaluating the summary
classification system using Deepeval framework.
"""
import json
from pathlib import Path
from typing import List, Dict, Any

# Deepeval imports with fallback
try:
    from deepeval.test_case import LLMTestCase
    from deepeval.dataset import EvaluationDataset
    HAS_DEEPEVAL = True
except ImportError:
    HAS_DEEPEVAL = False
    # Create dummy classes for when Deepeval is not available
    class LLMTestCase:
        def __init__(self, **kwargs):
            self.input = kwargs.get('input', '')
            self.actual_output = kwargs.get('actual_output', '')
            self.expected_output = kwargs.get('expected_output', '')
            self.context = kwargs.get('context', '')
    
    class EvaluationDataset:
        def __init__(self, **kwargs):
            pass


class SummaryClassificationTestCases:
    def __init__(self, data_path: str = None):
        self.data_path = data_path or str(Path(__file__).resolve().parents[2] / "data" / "summaries.json")
        self.test_cases = []
        self._load_test_cases()

    def _load_test_cases(self):
        """Load test cases from the summaries dataset."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                summaries_data = json.load(f)
            
            # Create test cases for each summary
            for item in summaries_data:
                summary = item.get('summary', '')
                doc_type = item.get('doc_type', '')
                doc_code = item.get('doc_code', '')
                
                # Create test case
                test_case = LLMTestCase(
                    input=summary,
                    actual_output="",  # Will be filled by the system
                    expected_output=doc_type,
                    context=summary,
                    metadata={
                        "doc_code": doc_code,
                        "doc_type": doc_type,
                        "summary_length": len(summary)
                    }
                )
                self.test_cases.append(test_case)
                
            print(f"✓ Loaded {len(self.test_cases)} test cases from {self.data_path}")
            
        except Exception as e:
            print(f"Error loading test cases: {e}")
            self.test_cases = []

    def get_test_cases(self, limit: int = None) -> List[LLMTestCase]:
        """Get test cases, optionally limited to a subset."""
        if limit:
            return self.test_cases[:limit]
        return self.test_cases

    def get_test_cases_by_type(self, doc_type: str) -> List[LLMTestCase]:
        """Get test cases filtered by document type."""
        return [tc for tc in self.test_cases if tc.metadata.get("doc_type") == doc_type]

    def create_evaluation_dataset(self, limit: int = None) -> EvaluationDataset:
        """Create a Deepeval EvaluationDataset from test cases."""
        test_cases = self.get_test_cases(limit)
        return EvaluationDataset(test_cases=test_cases)

    def get_sample_test_cases(self, num_samples: int = 10) -> List[LLMTestCase]:
        """Get a sample of test cases for quick testing."""
        import random
        return random.sample(self.test_cases, min(num_samples, len(self.test_cases)))


def create_custom_test_cases() -> List[LLMTestCase]:
    """Create custom test cases for specific evaluation scenarios."""
    custom_cases = [
        # Invoice test cases
        LLMTestCase(
            input="Payment of $1200 was made by ACME Corp on 2025-09-01 for invoice #INV-100.",
            actual_output="",
            expected_output="INVOICE",
            context="Payment of $1200 was made by ACME Corp on 2025-09-01 for invoice #INV-100.",
            metadata={"doc_type": "INVOICE", "test_type": "custom"}
        ),
        
        # Bank statement test cases
        LLMTestCase(
            input="The bank statement shows a withdrawal of $500 on 2025-08-30 from account 12345678.",
            actual_output="",
            expected_output="BANK_STATEMENT",
            context="The bank statement shows a withdrawal of $500 on 2025-08-30 from account 12345678.",
            metadata={"doc_type": "BANK_STATEMENT", "test_type": "custom"}
        ),
        
        # Leave request test cases
        LLMTestCase(
            input="John Doe submitted a leave request for 3 days starting 2025-09-10.",
            actual_output="",
            expected_output="LEAVE_REQUEST",
            context="John Doe submitted a leave request for 3 days starting 2025-09-10.",
            metadata={"doc_type": "LEAVE_REQUEST", "test_type": "custom"}
        ),
        
        # Purchase order test cases
        LLMTestCase(
            input="ACME Corp issued a purchase order #PO-2025-01 for 100 units of product X.",
            actual_output="",
            expected_output="PURCHASE_ORDER",
            context="ACME Corp issued a purchase order #PO-2025-01 for 100 units of product X.",
            metadata={"doc_type": "PURCHASE_ORDER", "test_type": "custom"}
        ),
        
        # Expense report test cases
        LLMTestCase(
            input="The expense report shows travel expenses of $350 for business trip to New York.",
            actual_output="",
            expected_output="EXPENSE_REPORT",
            context="The expense report shows travel expenses of $350 for business trip to New York.",
            metadata={"doc_type": "EXPENSE_REPORT", "test_type": "custom"}
        )
    ]
    
    return custom_cases


def create_edge_case_test_cases() -> List[LLMTestCase]:
    """Create edge case test cases for robustness testing."""
    edge_cases = [
        # Ambiguous cases
        LLMTestCase(
            input="Document contains financial information and payment details.",
            actual_output="",
            expected_output="UNKNOWN",  # Could be invoice, bank statement, or expense report
            context="Document contains financial information and payment details.",
            metadata={"doc_type": "UNKNOWN", "test_type": "edge_case", "difficulty": "ambiguous"}
        ),
        
        # Short summaries
        LLMTestCase(
            input="Invoice #123",
            actual_output="",
            expected_output="INVOICE",
            context="Invoice #123",
            metadata={"doc_type": "INVOICE", "test_type": "edge_case", "difficulty": "short"}
        ),
        
        # Long summaries
        LLMTestCase(
            input="This comprehensive document contains multiple sections including detailed financial transactions, payment schedules, vendor information, tax calculations, compliance notes, and administrative details that span several pages of complex business documentation.",
            actual_output="",
            expected_output="INVOICE",
            context="This comprehensive document contains multiple sections including detailed financial transactions, payment schedules, vendor information, tax calculations, compliance notes, and administrative details that span several pages of complex business documentation.",
            metadata={"doc_type": "INVOICE", "test_type": "edge_case", "difficulty": "long"}
        ),
        
        # Mixed content
        LLMTestCase(
            input="This document includes both invoice details for $500 and bank statement information showing account balance.",
            actual_output="",
            expected_output="MIXED",  # Contains multiple document types
            context="This document includes both invoice details for $500 and bank statement information showing account balance.",
            metadata={"doc_type": "MIXED", "test_type": "edge_case", "difficulty": "mixed"}
        )
    ]
    
    return edge_cases
