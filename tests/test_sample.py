from src.agents.crew_ai_agent import CrewAIAgent


def test_agent_runs_and_returns_expected_keys():
    agent = CrewAIAgent()
    summary = "Payment of $1200 was made by ACME Corp on 2025-09-01 for invoice #INV-100."
    result = agent.run(summary)
    assert isinstance(result, dict)
    assert "summary_type" in result
    assert "doc_code" in result
    assert "metrics" in result
    assert isinstance(result["metrics"], dict)
