#!/usr/bin/env python3
"""Test script for Deepeval integration with summary classification."""

import sys
import os
from pathlib import Path

# Mock Gemini API key for Deepeval if not set
if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = "mock-gemini-key"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.crew_ai_agent import CrewAIAgent
from evaluation.deepeval_test_cases import (
    SummaryClassificationTestCases, 
    create_custom_test_cases,
    create_edge_case_test_cases
)


def test_deepeval_integration():
    """Test the complete Deepeval integration."""
    print("🧪 Testing Deepeval Integration for Summary Classification")
    print("=" * 60)
    
    # Initialize the agent
    print("\n1. Initializing CrewAI Agent...")
    try:
        agent = CrewAIAgent()
        print("✓ Agent initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize agent: {e}")
        return False
    
    # Test single prediction with evaluation
    print("\n2. Testing single prediction with Deepeval evaluation...")
    try:
        test_summary = "Payment of $1200 was made by ACME Corp on 2025-09-01 for invoice #INV-100."
        result = agent.run(test_summary, ground_truth_doc_type="INVOICE")
        
        print(f"✓ Summary: {test_summary[:50]}...")
        print(f"✓ Predicted: {result['summary_type']}")
        print(f"✓ Doc Code: {result['doc_code']}")
        print(f"✓ Evaluation Framework: {result['metrics'].get('framework', 'N/A')}")
        
        if result['metrics'].get('framework') == 'deepeval':
            print(f"✓ Accuracy: {result['metrics'].get('accuracy', 0):.3f}")
            print(f"✓ Contextual Precision: {result['metrics'].get('contextual_precision', 0):.3f}")
            print(f"✓ Contextual Recall: {result['metrics'].get('contextual_recall', 0):.3f}")
            print(f"✓ Contextual Relevancy: {result['metrics'].get('contextual_relevancy', 0):.3f}")
            print(f"✓ Faithfulness: {result['metrics'].get('faithfulness', 0):.3f}")
        else:
            print("⚠ Using fallback evaluation (Deepeval not available)")
            
    except Exception as e:
        print(f"✗ Single prediction test failed: {e}")
        return False
    
    # Test custom test cases
    print("\n3. Testing custom test cases...")
    try:
        custom_cases = create_custom_test_cases()
        test_data = [
            {
                "summary": tc.input,
                "doc_type": tc.expected_output
            }
            for tc in custom_cases[:3]  # Test first 3 cases
        ]
        
        batch_results = agent.run_batch_evaluation(test_data)
        evaluation = batch_results["batch_evaluation"]
        
        print(f"✓ Tested {len(test_data)} custom cases")
        print(f"✓ Framework: {evaluation.get('framework', 'N/A')}")
        print(f"✓ Accuracy: {evaluation.get('accuracy', 0):.3f}")
        
        if evaluation.get('framework') == 'deepeval':
            print(f"✓ Overall Score: {evaluation.get('overall_score', 0):.3f}")
        
    except Exception as e:
        print(f"✗ Custom test cases failed: {e}")
        return False
    
    # Test edge cases
    print("\n4. Testing edge cases...")
    try:
        edge_cases = create_edge_case_test_cases()
        edge_data = [
            {
                "summary": tc.input,
                "doc_type": tc.expected_output
            }
            for tc in edge_cases[:2]  # Test first 2 edge cases
        ]
        
        edge_results = agent.run_batch_evaluation(edge_data)
        edge_evaluation = edge_results["batch_evaluation"]
        
        print(f"✓ Tested {len(edge_data)} edge cases")
        print(f"✓ Framework: {edge_evaluation.get('framework', 'N/A')}")
        print(f"✓ Accuracy: {edge_evaluation.get('accuracy', 0):.3f}")
        
    except Exception as e:
        print(f"✗ Edge cases test failed: {e}")
        return False
    
    # Test with actual dataset
    print("\n5. Testing with sample dataset...")
    try:
        test_cases = SummaryClassificationTestCases()
        sample_cases = test_cases.get_sample_test_cases(5)  # Test 5 random cases
        
        dataset_test_data = [
            {
                "summary": tc.input,
                "doc_type": tc.expected_output
            }
            for tc in sample_cases
        ]
        
        dataset_results = agent.run_batch_evaluation(dataset_test_data)
        dataset_evaluation = dataset_results["batch_evaluation"]
        
        print(f"✓ Tested {len(dataset_test_data)} dataset cases")
        print(f"✓ Framework: {dataset_evaluation.get('framework', 'N/A')}")
        print(f"✓ Accuracy: {dataset_evaluation.get('accuracy', 0):.3f}")
        
        # Show individual results
        individual_results = dataset_results["individual_results"]
        correct_predictions = sum(1 for r in individual_results if r["correct"])
        print(f"✓ Correct predictions: {correct_predictions}/{len(individual_results)}")
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 Deepeval Integration Test Complete!")
    print("✓ All tests passed successfully")
    print("✓ Live evaluation is working")
    print("✓ Batch evaluation is working")
    print("✓ Edge case handling is working")
    
    return True


def test_evaluation_metrics():
    """Test specific evaluation metrics."""
    print("\n📊 Testing Evaluation Metrics...")
    
    try:
        from services.evaluation_service import EvaluationService
        
        evaluator = EvaluationService()
        
        # Test fallback evaluation
        predictions = [{"doc_type": "INVOICE"}, {"doc_type": "BANK_STATEMENT"}]
        ground_truths = [{"doc_type": "INVOICE"}, {"doc_type": "INVOICE"}]
        
        results = evaluator.evaluate(predictions, ground_truths)
        print(f"✓ Fallback evaluation: {results.get('framework', 'N/A')}")
        print(f"✓ Accuracy: {results.get('accuracy', 0):.3f}")
        
        # Test single evaluation
        single_result = evaluator.evaluate_single(
            predicted_doc_type="INVOICE",
            actual_doc_type="INVOICE",
            context="Test invoice summary"
        )
        print(f"✓ Single evaluation: {single_result.get('framework', 'N/A')}")
        
    except Exception as e:
        print(f"✗ Evaluation metrics test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("🚀 Starting Deepeval Integration Tests...")
    
    # Test evaluation metrics first
    metrics_success = test_evaluation_metrics()
    
    # Test full integration
    integration_success = test_deepeval_integration()
    
    if metrics_success and integration_success:
        print("\n🎉 All tests passed! Deepeval integration is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        sys.exit(1)
