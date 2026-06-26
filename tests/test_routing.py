"""Tests for triage routing decisions."""

from unittest.mock import MagicMock

import pytest

from agents.triage_agent import TriageAgent, CONFIDENCE_ESCALATION_THRESHOLD
from models import AgentResponse, ConversationState, Message, MessageRole
from conftest import FAST_MODEL, make_triage_json_response


def _triage_with_mock_groq(mock_groq: MagicMock) -> TriageAgent:
    return TriageAgent(
        config={},
        retriever=MagicMock(),
        groq_client=mock_groq,
    )


def test_billing_query_routes_to_billing_agent(mock_groq: MagicMock) -> None:
    mock_groq.chat_completion_json.return_value = make_triage_json_response(
        intent="billing",
        target_agent="Billing",
        confidence=0.92,
        entities={"plan_type": "Pro"},
    )
    agent = _triage_with_mock_groq(mock_groq)
    state = ConversationState()
    state.messages.append(
        Message(role=MessageRole.USER, content="I have a question about my invoice")
    )

    result = agent._classify(state, "I have a question about my invoice")
    response = agent.run(state)

    assert result.intent == "billing"
    assert result.target_agent == "Billing"
    assert response.handover_target == "Billing"
    assert response.requires_handover is True
    assert mock_groq.chat_completion_json.call_count >= 1
    assert mock_groq.chat_completion_json.call_args.kwargs["model"] == FAST_MODEL


def test_technical_query_routes_to_technical_agent(mock_groq: MagicMock) -> None:
    mock_groq.chat_completion_json.return_value = make_triage_json_response(
        intent="technical",
        target_agent="TechnicalSupport",
        confidence=0.88,
        entities={"product_area": "alerts"},
        reasoning="Alert monitoring issue",
    )
    agent = _triage_with_mock_groq(mock_groq)
    state = ConversationState()
    state.messages.append(
        Message(role=MessageRole.USER, content="My alerts stopped firing")
    )

    result = agent._classify(state, "My alerts stopped firing")
    response = agent.run(state)

    assert result.intent == "technical"
    assert result.target_agent == "TechnicalSupport"
    assert response.handover_target == "TechnicalSupport"


def test_low_confidence_routes_to_escalation(mock_groq: MagicMock) -> None:
    mock_groq.chat_completion_json.return_value = make_triage_json_response(
        intent="unknown",
        target_agent="TechnicalSupport",
        confidence=0.4,
        reasoning="Unclear customer request",
    )
    agent = _triage_with_mock_groq(mock_groq)
    state = ConversationState()

    result = agent._classify(state, "something vague and unclear")

    assert result.confidence < CONFIDENCE_ESCALATION_THRESHOLD
    assert result.target_agent == "Escalation"


def test_unknown_intent_routes_to_triage(mock_groq: MagicMock) -> None:
    mock_groq.chat_completion_json.return_value = make_triage_json_response(
        intent="unknown",
        target_agent="Triage",
        confidence=0.75,
        reasoning="Gibberish detected",
    )
    agent = _triage_with_mock_groq(mock_groq)
    state = ConversationState()
    state.messages.append(
        Message(role=MessageRole.USER, content="xyzzy plugh qwerty asdfg")
    )

    response = agent.run(state)

    assert isinstance(response, AgentResponse)
    assert response.agent_name == "Triage"
    assert response.content
    assert 0.0 <= response.confidence <= 1.0
