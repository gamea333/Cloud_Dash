"""Tests for escalation agent priority and packaging."""

from agents.escalation_agent import EscalationAgent
from models import ConversationState, ConversationStatus, Message, MessageRole


class MockRetriever:
    def retrieve(self, **kwargs):
        return []


class MockGroq:
    def chat_completion(self, **kwargs):
        return ""


def _escalation_agent() -> EscalationAgent:
    return EscalationAgent(
        config={},
        retriever=MockRetriever(),  # type: ignore[arg-type]
        groq_client=MockGroq(),  # type: ignore[arg-type]
    )


def test_critical_priority_for_legal_keywords():
    agent = _escalation_agent()
    state = ConversationState()
    state.messages.append(
        Message(role=MessageRole.USER, content="I will contact my lawyer about this charge")
    )
    package = agent._build_escalation_package(state)
    assert package.priority == "critical"
    assert package.sentiment in ("neutral", "angry", "frustrated")


def test_marks_conversation_escalated():
    agent = _escalation_agent()
    state = ConversationState()
    state.messages.append(
        Message(role=MessageRole.USER, content="escalate this urgent issue")
    )
    response = agent.run(state)
    assert state.status == ConversationStatus.ESCALATED
    assert "HUMAN OPERATOR ALERT" in response.content


def test_human_alert_format():
    agent = _escalation_agent()
    state = ConversationState()
    state.extracted_entities["customer_id"] = "cust-test-123"
    state.messages.append(
        Message(role=MessageRole.USER, content="This is unacceptable, I need a manager")
    )
    package = agent._build_escalation_package(state)
    alert = agent._format_human_alert(package)
    assert "HUMAN OPERATOR ALERT" in alert
    assert "cust-test-123" in alert
