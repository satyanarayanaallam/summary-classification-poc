# Summary Classification POC (Triplet-Based Retrieval + CrewAI Agent + Deepeval)

## Project Overview
This proof-of-concept project performs **summary classification** using **triplet-based retrieval** and **live evaluation** with Deepeval, orchestrated through **CrewAI** agents.  

- Input: Document summaries (instead of full PDFs)  
- Output: Classified **summary type** and **document code**  
- CrewAI orchestrates the pipeline and handles agentic decision-making, error handling, and live evaluation metrics.

---

## Architecture
[User Summary Input]
│
▼
[CrewAI Orchestrator Agent]
│
├──> Triplet Extraction Agent (Gemini)
│
├──> Triplet Normalization / Masking Agent
│
├──> Vector DB Storage Agent
│
├──> Retrieval Agent (Query → Vector DB → Summary Type & Code)
│
└──> Deepeval Evaluation Agent (Live Metrics)


---

## Features
- **Triplet extraction** from summaries using Gemini Flash
- **PII masking & normalization** (common nouns only)
- **Vector DB storage and retrieval**
- **Live evaluation** using Deepeval
- **Agentic orchestration** using CrewAI
- Modular architecture following **SOLID principles** and clean code practices

---

## Project Structure



src/
├── agents/ # CrewAI orchestrator and agent scaffolds
│ └── crew_ai_agent.py
├── config/ # Environment & settings
├── models/ # LLM clients (Gemini)
├── db/ # Vector DB client
├── services/ # Triplet service, retrieval, evaluation
├── utils/ # Normalization, logging
├── main.py # Example POC flow (single summary run)
data/ # Sample summaries with types and codes
tests/ # Unit tests
.env # API keys


---

## Sample Data

- `data/summaries.json` contains summary text, summary type (`doc_type`), and summary code (`doc_code`):

```json
[
  {
    "summary": "Payment of $1200 was made by ACME Corp on 2025-09-01 for invoice #INV-100.",
    "doc_type": "INVOICE",
    "doc_code": "INV001"
  },
  {
    "summary": "The bank statement shows a withdrawal of $500 on 2025-08-30 from account 12345678.",
    "doc_type": "BANK_STATEMENT",
    "doc_code": "BS001"
  },
  {
    "summary": "John Doe submitted a leave request for 3 days starting 2025-09-10.",
    "doc_type": "LEAVE_REQUEST",
    "doc_code": "LR001"
  }
]

Installation

Clone the repo

git clone <repo_url>
cd summary-classification-poc


Create virtual environment

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


Install dependencies

pip install -r requirements.txt


Create .env file

GEMINI_API_KEY=<your_gemini_api_key>
VECTOR_DB_API_KEY=<your_vector_db_key>

Usage (POC with CrewAI)
from src.agents.crew_ai_agent import CrewAIAgent

# Example summary
summary = "ACME Corp issued a purchase order #PO-2025-01 for 100 units of product X."

# Initialize CrewAI orchestrator agent
agent = CrewAIAgent()

# Run the POC through CrewAI agent
result = agent.run(summary)

print("Summary Type:", result['summary_type'])
print("Document Code:", result['doc_code'])
print("Evaluation Metrics:", result['metrics'])


The CrewAI agent:

Extracts triplets from summary

Normalizes & masks triplets

Stores triplets in vector DB

Queries a sample triplet

Retrieves summary type & doc code

Evaluates correctness via Deepeval

Returns structured result with metrics

Testing

Run all unit tests:

pytest tests/

Future Improvements

Integrate full PDF ingestion & OCR

Combine semantic summary search (E1) with triplet retrieval (E2) for hybrid evaluation

Fine-tune triplet extraction prompts for domain-specific accuracy

Add confidence scoring & threshold-based classification fallback

Extend CrewAI orchestration for multi-agent workflows and auto-retries


Notes:
- The `GeminiClient` is a simple heuristic extractor (no external LLM calls).
- `VectorDBClient` is an in-memory placeholder for demo purposes.
- Replace stubs and add `.env` keys to integrate real services.

## How to run (Windows PowerShell)

1. Create a virtual environment and activate it:

```powershell
python -m venv venv
; .\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the example POC flow:

```powershell
python -m src.main
```

4. Run tests:

```powershell
python -m pytest -q
```
