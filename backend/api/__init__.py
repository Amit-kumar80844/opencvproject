"""
backend/api/__init__.py
BSc Data Science Capstone — NIBM, 2026
Author : Manula Fernando

Public API for the agentic layer:
  - rag_store    : ChromaDB vector store for agronomy guidelines
  - agent_graph  : LangGraph multi-agent treatment recommendation pipeline
"""
from backend.api.rag_store import (
    get_or_create_store,
    ingest_guidelines,
    query_guidelines,
    AGRONOMY_GUIDELINES,
    COLLECTION_NAME,
)
from backend.api.agent_graph import (
    run_agent_pipeline,
    build_agent_graph,
    AgentState,
)

__all__ = [
    "get_or_create_store",
    "ingest_guidelines",
    "query_guidelines",
    "AGRONOMY_GUIDELINES",
    "COLLECTION_NAME",
    "run_agent_pipeline",
    "build_agent_graph",
    "AgentState",
]
