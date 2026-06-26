"""Tests for RAG retrieval, query rewriting, and no-result escalation paths."""

from unittest.mock import MagicMock, patch

import pytest

from agents.technical_agent import TechnicalSupportAgent
from models import ConversationState, Message, MessageRole
from retrieval.retriever import KnowledgeRetriever
from conftest import FAST_MODEL


@pytest.fixture
def indexed_retriever() -> KnowledgeRetriever:
    """Use real ChromaDB index when available (populated by ingest)."""
    mock_groq = MagicMock()
    mock_groq.chat_completion.return_value = "AWS CloudWatch integration stopped working"
    retriever = KnowledgeRetriever(groq_client=mock_groq)
    retriever.ensure_indexed()
    return retriever


def test_kb_retrieval_returns_relevant_doc(indexed_retriever: KnowledgeRetriever) -> None:
    chunks = indexed_retriever.retrieve(
        query="alerts not firing",
        top_k=3,
        conversation_history=[],
        rewrite=False,
    )
    assert len(chunks) >= 1
    categories = {c.get("category", "") for c in chunks}
    article_ids = {c.get("article_id", "") for c in chunks}
    assert "troubleshooting" in categories or "KB-005" in article_ids


def test_query_rewriter_uses_conversation_context() -> None:
    mock_groq = MagicMock()
    mock_groq.chat_completion.return_value = "AWS CloudWatch integration stopped working"

    retriever = KnowledgeRetriever(groq_client=mock_groq)
    history = [
        Message(
            role=MessageRole.USER,
            content="My AWS CloudWatch integration was working yesterday",
        ),
        Message(
            role=MessageRole.ASSISTANT,
            content="Let me help with your AWS integration.",
            agent_name="TechnicalSupport",
        ),
    ]

    rewritten = retriever.query_rewriter("it stopped working", history)

    mock_groq.chat_completion.assert_called_once()
    assert mock_groq.chat_completion.call_args.kwargs["model"] == FAST_MODEL
    assert "AWS" in rewritten or "integration" in rewritten.lower()


def test_no_results_triggers_escalation_offer() -> None:
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {
            "article_id": "KB-999",
            "title": "Unrelated",
            "content": "Unrelated content",
            "distance": 0.95,
        }
    ]
    mock_retriever.format_citations.return_value = "Source: KB-999 | Unrelated"

    mock_groq = MagicMock()
    agent = TechnicalSupportAgent(
        config={"routing_rules": {"can_handover_to": ["Billing", "Escalation"]}},
        retriever=mock_retriever,
        groq_client=mock_groq,
    )
    state = ConversationState()
    state.messages.append(
        Message(role=MessageRole.USER, content="quantum flux capacitor monitoring")
    )

    response = agent.run(state)

    combined = response.content.lower()
    assert "escalate" in combined or "verified article" in combined


def test_format_citations_empty() -> None:
    assert KnowledgeRetriever.format_citations([]) == ""


def test_format_citations_deduplicates() -> None:
    chunks = [
        {"article_id": "KB-001", "title": "API Key Reset"},
        {"article_id": "KB-001", "title": "API Key Reset"},
        {"article_id": "KB-002", "title": "Cloud Providers"},
    ]
    result = KnowledgeRetriever.format_citations(chunks)
    assert "Source: KB-001 | API Key Reset" in result
    assert "Source: KB-002 | Cloud Providers" in result
    assert result.count("KB-001") == 1
