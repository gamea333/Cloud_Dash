"""Shared pytest fixtures and mocks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.groq_client import DEFAULT_MODEL, FAST_MODEL
from utils.logger import configure_logging


@pytest.fixture(autouse=True)
def _configure_test_logging() -> None:
    configure_logging("WARNING")


@pytest.fixture
def mock_groq() -> MagicMock:
    """Mock Groq client; tests patch methods per scenario."""
    client = MagicMock()
    client.chat_completion_json.return_value = {
        "intent": "unknown",
        "entities": {},
        "target_agent": "TechnicalSupport",
        "confidence": 0.5,
        "reasoning": "Default mock classification",
    }
    client.chat_completion.return_value = "Mock LLM response for testing."
    return client


@pytest.fixture
def mock_retriever() -> MagicMock:
    """Mock knowledge retriever with format_citations support."""
    from retrieval.retriever import KnowledgeRetriever

    retriever = MagicMock()
    retriever.retrieve.return_value = []
    retriever.format_citations.side_effect = KnowledgeRetriever.format_citations
    retriever.query_rewriter.side_effect = lambda query, history: query
    return retriever


def make_triage_json_response(
    intent: str,
    target_agent: str,
    confidence: float = 0.9,
    entities: dict[str, Any] | None = None,
    reasoning: str = "Test classification",
) -> dict[str, Any]:
    return {
        "intent": intent,
        "entities": entities or {},
        "target_agent": target_agent,
        "confidence": confidence,
        "reasoning": reasoning,
    }


__all__ = ["DEFAULT_MODEL", "FAST_MODEL", "make_triage_json_response", "mock_groq", "mock_retriever"]
