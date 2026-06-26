"""Tests for handover protocol and orchestrator fallback."""

from unittest.mock import MagicMock, patch

import pytest

from agents.base import BaseAgent
from agents.orchestrator import Orchestrator
from handover.handover_manager import HandoverManager
from models import AgentResponse, ConversationState, HandoverLog, HandoverPayload, Message, MessageRole
from utils.logger import SupportLogger


class StubAgent(BaseAgent):
    name = "Stub"

    def __init__(self, name: str = "Stub", **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.name = name

    def run(self, conversation_state: ConversationState) -> AgentResponse:
        return AgentResponse(content=f"{self.name} handled", agent_name=self.name)


class FailingRunAgent(StubAgent):
    def run(self, conversation_state: ConversationState) -> AgentResponse:
        raise RuntimeError("Simulated agent failure")


def _make_manager(agent_names: list[str]) -> HandoverManager:
    agents: dict[str, BaseAgent] = {}
    for name in agent_names:
        agents[name] = StubAgent(
            name=name,
            config={},
            retriever=MagicMock(),
            groq_client=MagicMock(),
        )
    return HandoverManager(agents)


@pytest.fixture
def logger() -> SupportLogger:
    state = ConversationState()
    return SupportLogger(trace_id=state.trace_id, conversation_id=state.conversation_id)


def test_handover_preserves_entities(logger: SupportLogger) -> None:
    manager = _make_manager(["Triage", "Billing"])
    state = ConversationState()
    state.extracted_entities["customer_id"] = "CUST-123"

    payload = HandoverPayload(
        source_agent="Triage",
        target_agent="Billing",
        reason="Billing inquiry",
        conversation_summary="Invoice question",
        extracted_entities={"customer_id": "CUST-123", "plan_type": "Pro"},
    )
    success = manager.execute_handover(state, payload, logger)

    assert success is True
    assert state.extracted_entities["customer_id"] == "CUST-123"
    assert state.current_agent == "Billing"


def test_failed_handover_falls_back_to_triage(logger: SupportLogger) -> None:
    manager = _make_manager(["Triage", "Billing"])
    state = ConversationState(current_agent="Triage")
    state.messages.append(Message(role=MessageRole.USER, content="billing help"))

    payload = HandoverPayload(
        source_agent="Triage",
        target_agent="Billing",
        reason="Billing inquiry",
        conversation_summary="test",
    )

    with patch.object(HandoverLog, "__init__", side_effect=RuntimeError("handover log failure")):
        success = manager.execute_handover(state, payload, logger)

    assert success is False
    assert state.current_agent == "Triage"


def test_orchestrator_falls_back_to_triage_on_failed_handover() -> None:
    mock_groq = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_retriever.format_citations.return_value = ""

    agents: dict[str, BaseAgent] = {
        "Triage": StubAgent(
            name="Triage",
            config={},
            retriever=mock_retriever,
            groq_client=mock_groq,
        ),
        "Billing": StubAgent(
            name="Billing",
            config={},
            retriever=mock_retriever,
            groq_client=mock_groq,
        ),
    }
    triage_agent = agents["Triage"]

    def triage_run(state: ConversationState) -> AgentResponse:
        return AgentResponse(
            content="Routing to billing",
            agent_name="Triage",
            requires_handover=True,
            handover_target="Billing",
            confidence=0.9,
        )

    triage_agent.run = triage_run  # type: ignore[method-assign]

    handover_manager = HandoverManager(agents)
    orchestrator = Orchestrator(mock_groq, mock_retriever, handover_manager)

    state = ConversationState(current_agent="Triage")

    with patch.object(handover_manager, "execute_handover", return_value=False) as mock_handover:
        with patch.object(orchestrator, "_run_agent", wraps=orchestrator._run_agent) as mock_run:
            response = orchestrator.route(state, "invoice question")

    mock_handover.assert_called()
    assert mock_run.call_args_list[0][0][0] == "Triage"
    assert isinstance(response, AgentResponse)


def test_handover_log_entry_created(logger: SupportLogger) -> None:
    manager = _make_manager(["Triage", "TechnicalSupport"])
    state = ConversationState()
    state.messages.append(
        Message(role=MessageRole.USER, content="My alerts stopped firing")
    )

    payload = HandoverPayload(
        source_agent="Triage",
        target_agent="TechnicalSupport",
        reason="Technical issue",
        conversation_summary="Alert problem",
        extracted_entities={"product_area": "alerts"},
    )
    manager.execute_handover(state, payload, logger)

    logs = manager.get_handover_log(state.conversation_id)
    assert len(logs) == 1
    entry = logs[0]
    assert entry.handover_payload.source_agent == "Triage"
    assert entry.handover_payload.target_agent == "TechnicalSupport"
    assert entry.handover_payload.timestamp is not None


def test_execute_handover_unknown_agent_falls_back(logger: SupportLogger) -> None:
    manager = _make_manager(["Triage"])
    state = ConversationState(current_agent="TechnicalSupport")

    payload = HandoverPayload(
        source_agent="TechnicalSupport",
        target_agent="Billing",
        reason="Billing issue",
        conversation_summary="test",
    )
    success = manager.execute_handover(state, payload, logger)

    assert success is False
    assert state.current_agent == "Triage"
