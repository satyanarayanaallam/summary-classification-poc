"""Example POC flow (single summary run) as described in README."""
from src.agents.crew_ai_agent import CrewAIAgent


def main():
    summary = "ACME Corp issued a purchase order #PO-2025-01 for 100 units of product X."
    agent = CrewAIAgent()
    result = agent.run(summary)
    print("Summary Type:", result["summary_type"])
    print("Document Code:", result["doc_code"])
    print("Evaluation Metrics:", result["metrics"])


if __name__ == "__main__":
    main()
