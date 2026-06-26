"""Handover protocol for agent-to-agent transfers."""

from typing import Any, Optional

from models import (
    ConversationState,
    ConversationStatus,
    HandoverLog,
    HandoverPayload,
    Message,
    MessageRole,
)
from agents.base import BaseAgent
from utils.logger import SupportLogger


class HandoverManager:
    """Executes validated handovers with logging and fallback to Triage."""

    def __init__(self, agents: dict[str, BaseAgent]) -> None:
        self.agents = agents
        self.handover_logs: dict[str, list[HandoverLog]] = {}

    def _agent_exists(self, agent_name: str) -> bool:
        return agent_name in self.agents

    def _summarize_last_agent_work(self, state: ConversationState) -> str:
        for msg in reversed(state.messages):
            if msg.role == MessageRole.ASSISTANT and msg.agent_name:
                return f"{msg.agent_name}: {msg.content[:500]}"
        return "No prior agent response recorded."

    def _build_context_snapshot(self, state: ConversationState) -> dict[str, Any]:
        return {
            "message_count": len(state.messages),
            "messages": [
                {
                    "role": m.role.value,
                    "content": m.content[:300],
                    "agent_name": m.agent_name,
                }
                for m in state.messages
            ],
            "current_agent": state.current_agent,
            "status": state.status.value,
            "entities": dict(state.extracted_entities),
            "last_agent_work": self._summarize_last_agent_work(state),
        }

    def execute_handover(
        self,
        conversation_state: ConversationState,
        handover_payload: HandoverPayload,
        logger: SupportLogger,
    ) -> bool:
        """
        Validate target agent, update state, and log the handover.
        Returns True on success, False on failure (falls back to Triage).
        """
        source = handover_payload.source_agent
        target = handover_payload.target_agent

        logger.handover_initiated(
            source_agent=source,
            target_agent=target,
            reason=handover_payload.reason,
            priority=handover_payload.priority,
        )

        if not self._agent_exists(target):
            logger.handover_failed(
                source_agent=source,
                target_agent=target,
                error=f"Target agent '{target}' not found",
            )
            conversation_state.current_agent = "Triage"
            conversation_state.status = ConversationStatus.ACTIVE
            return False

        try:
            conversation_state.current_agent = target
            conversation_state.extracted_entities.update(handover_payload.extracted_entities)
            conversation_state.status = ConversationStatus.HANDOVER

            log_entry = HandoverLog(
                trace_id=conversation_state.trace_id,
                handover_payload=handover_payload,
                context_snapshot=self._build_context_snapshot(conversation_state),
            )
            if conversation_state.conversation_id not in self.handover_logs:
                self.handover_logs[conversation_state.conversation_id] = []
            self.handover_logs[conversation_state.conversation_id].append(log_entry)

            conversation_state.status = ConversationStatus.ACTIVE
            logger.handover_completed(source_agent=source, target_agent=target)
            return True

        except Exception as exc:
            logger.handover_failed(
                source_agent=source,
                target_agent=target,
                error=str(exc),
            )
            conversation_state.current_agent = "Triage"
            conversation_state.status = ConversationStatus.ACTIVE
            return False

    def get_handover_log(self, conversation_id: str) -> list[HandoverLog]:
        return self.handover_logs.get(conversation_id, [])

    def add_assistant_message(
        self,
        state: ConversationState,
        content: str,
        agent_name: str,
    ) -> Message:
        msg = Message(
            role=MessageRole.ASSISTANT,
            content=content,
            agent_name=agent_name,
        )
        state.messages.append(msg)
        return msg

    def build_payload_from_response(
        self,
        state: ConversationState,
        source_agent: str,
        target_agent: str,
        reason: str = "Agent-initiated handover",
    ) -> HandoverPayload:
        pending = state.extracted_entities.get("pending_handover")
        if isinstance(pending, dict) and pending.get("target_agent") == target_agent:
            return HandoverPayload(**pending)

        return HandoverPayload(
            source_agent=source_agent,
            target_agent=target_agent,
            reason=reason,
            conversation_summary=self._build_context_snapshot(state)["last_agent_work"],
            extracted_entities=dict(state.extracted_entities),
            priority="normal",
        )
