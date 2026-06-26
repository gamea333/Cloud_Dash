"""Tests for billing agent mock account lookup and escalation triggers."""

from agents.billing_agent import BillingAgent, REFUND_AUTHORITY_LIMIT
from models import ConversationState, Message, MessageRole


class MockRetriever:
    def retrieve(self, **kwargs):
        return []


class MockGroq:
    def chat_completion(self, **kwargs):
        return "Your billing question has been reviewed."


def _billing_agent() -> BillingAgent:
    return BillingAgent(
        config={"routing_rules": {"can_handover_to": ["Escalation", "TechnicalSupport"]}},
        retriever=MockRetriever(),  # type: ignore[arg-type]
        groq_client=MockGroq(),  # type: ignore[arg-type]
    )


def test_mock_account_lookup_deterministic():
    agent = _billing_agent()
    account_a = agent._mock_account_lookup("cust-acme-001")
    account_b = agent._mock_account_lookup("cust-acme-001")
    assert account_a == account_b
    assert account_a["customer_id"] == "cust-acme-001"
    assert account_a["plan"] in ("Starter", "Pro", "Enterprise")


def test_escalation_on_manager_request():
    agent = _billing_agent()
    state = ConversationState()
    state.messages.append(
        Message(role=MessageRole.USER, content="I need to speak to a manager about my bill")
    )
    response = agent.run(state)
    assert response.requires_handover is True
    assert response.handover_target == "Escalation"


def test_escalation_on_large_refund():
    agent = _billing_agent()
    needs, reason = agent._needs_escalation(
        f"I want a $600 refund immediately",
        {"last_invoice": {"amount": 199.0}},
    )
    assert needs is True
    assert str(int(REFUND_AUTHORITY_LIMIT)) in reason or "600" in reason
