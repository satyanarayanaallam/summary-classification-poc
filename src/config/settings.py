"""Configuration settings for the summary-classification POC.

This module centralizes configuration and environment variable access.
Keep keys out of source control and populate a `.env` file for local runs.
"""
import os
from typing import Optional


OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")

# Vector store settings
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "384"))  # Default for all-MiniLM-L6-v2
VECTOR_MODEL_NAME = os.getenv("VECTOR_MODEL_NAME", "all-MiniLM-L6-v2")

# CrewAI / orchestrator settings
CREWAI_AGENT_NAME = os.getenv("CREWAI_AGENT_NAME", "summary-orchestrator")

# Deepeval settings
DEEPEVAL_ENABLED = os.getenv("DEEPEVAL_ENABLED", "true").lower() == "true"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Required for Deepeval with Gemini
DEEPEVAL_MODEL = os.getenv("DEEPEVAL_MODEL", "gemini-1.5-flash")
