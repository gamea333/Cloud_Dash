"""Tests for technical agent KB threshold behavior."""

from agents.technical_agent import TechnicalSupportAgent, SIMILARITY_THRESHOLD


class MockRetriever:
    def retrieve(self, **kwargs):
        return [
            {
                "article_id": "KB-005",
                "title": "Alerts Not Firing",
                "content": "Alert troubleshooting content",
                "distance": 0.8,
            }
        ]

    @staticmethod
    def format_citations(chunks):
        return "Source: KB-005 | Alerts Not Firing"


class MockGroq:
    def chat_completion(self, **kwargs):
        return "Should not be called"


def test_low_similarity_returns_no_fabrication_response():
    agent = TechnicalSupportAgent(
        config={"routing_rules": {"can_handover_to": ["Billing", "Escalation"]}},
        retriever=MockRetriever(),  # type: ignore[arg-type]
        groq_client=MockGroq(),  # type: ignore[arg-type]
    )
    from models import ConversationState, Message, MessageRole

    state = ConversationState()
    state.messages.append(
        Message(role=MessageRole.USER, content="My alerts are not firing")
    )
    response = agent.run(state)
    assert "couldn't find a verified article" in response.content
    assert "escalate" in response.content.lower()


def test_similarity_threshold_filters_chunks():
    agent = TechnicalSupportAgent(
        config={},
        retriever=MockRetriever(),  # type: ignore[arg-type]
        groq_client=MockGroq(),  # type: ignore[arg-type]
    )
    chunks = [{"distance": 0.8}, {"distance": 0.3}]
    relevant = agent._filter_relevant_chunks(chunks)
    assert len(relevant) == 1
    assert 1.0 - 0.3 >= SIMILARITY_THRESHOLD
