"""Tests for CloudDash Pydantic models."""

from models import (
    AgentResponse,
    ConversationState,
    ConversationStatus,
    EscalationPackage,
    HandoverLog,
    HandoverPayload,
    Message,
    MessageRole,
)


def test_message_defaults():
    msg = Message(role=MessageRole.USER, content="Hello")
    assert msg.role == MessageRole.USER
    assert msg.content == "Hello"
    assert msg.agent_name is None
    assert msg.timestamp is not None


def test_conversation_state_defaults():
    state = ConversationState()
    assert state.conversation_id
    assert state.trace_id
    assert state.messages == []
    assert state.current_agent == "Triage"
    assert state.status == ConversationStatus.ACTIVE


def test_agent_response_handover_fields():
    response = AgentResponse(
        content="Routing to billing",
        agent_name="Triage",
        requires_handover=True,
        handover_target="Billing",
        confidence=0.9,
        kb_sources_cited=["KB-009"],
    )
    assert response.requires_handover is True
    assert response.handover_target == "Billing"
    assert "KB-009" in response.kb_sources_cited


def test_handover_payload():
    payload = HandoverPayload(
        source_agent="Triage",
        target_agent="TechnicalSupport",
        reason="Technical issue detected",
        conversation_summary="User reports alerts not firing",
    )
    assert payload.priority == "normal"
    assert payload.timestamp is not None


def test_handover_log():
    payload = HandoverPayload(
        source_agent="Triage",
        target_agent="Billing",
        reason="Billing question",
        conversation_summary="Invoice inquiry",
    )
    log = HandoverLog(trace_id="trace-123", handover_payload=payload)
    assert log.trace_id == "trace-123"
    assert log.handover_payload.source_agent == "Triage"


def test_escalation_package():
    package = EscalationPackage(
        conversation_id="conv-123",
        summary="Customer frustrated with outage",
        priority="critical",
        sentiment="negative",
        issue_type="escalation",
    )
    assert package.priority == "critical"
    assert package.sentiment == "negative"
