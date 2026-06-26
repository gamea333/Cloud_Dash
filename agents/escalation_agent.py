"""Escalation agent — packages cases for human operators."""

import re
from typing import Any

from models import (
    AgentResponse,
    ConversationState,
    ConversationStatus,
    EscalationPackage,
    MessageRole,
)
from agents.base import BaseAgent

CRITICAL_KEYWORDS = ["lawyer", "legal", "chargeback", "sue", "lawsuit", "attorney"]
HIGH_KEYWORDS = ["urgent", "manager", "supervisor", "immediately", "asap", "critical"]
FRUSTRATED_KEYWORDS = ["frustrated", "disappointed", "unacceptable", "ridiculous", "terrible"]
ANGRY_KEYWORDS = ["furious", "angry", "hate", "worst", "useless", "incompetent"]


class EscalationAgent(BaseAgent):
    """Summarizes conversations and simulates human operator handover."""

    name = "Escalation"

    def _count_resolution_attempts(self, state: ConversationState) -> int:
        attempts = int(state.extracted_entities.get("resolution_attempts", 0))
        user_msgs = [m for m in state.messages if m.role == MessageRole.USER]
        retry_signals = ["still not", "didn't work", "not helpful", "same issue", "again", "escalate"]
        for msg in user_msgs:
            if any(s in msg.content.lower() for s in retry_signals):
                attempts += 1
        return max(attempts, len([m for m in state.messages if m.role == MessageRole.ASSISTANT]) - 1)

    def _detect_sentiment(self, state: ConversationState) -> str:
        user_text = " ".join(
            m.content.lower() for m in state.messages if m.role == MessageRole.USER
        )
        if any(w in user_text for w in ANGRY_KEYWORDS):
            return "angry"
        if any(w in user_text for w in FRUSTRATED_KEYWORDS):
            return "frustrated"
        positive = ["thanks", "thank you", "helpful", "great"]
        if any(w in user_text for w in positive):
            return "positive"
        return "neutral"

    def _assign_priority(
        self,
        state: ConversationState,
        sentiment: str,
        resolution_attempts: int,
    ) -> str:
        combined = " ".join(m.content.lower() for m in state.messages)
        if any(kw in combined for kw in CRITICAL_KEYWORDS):
            return "critical"
        if any(kw in combined for kw in HIGH_KEYWORDS):
            return "high"
        if sentiment in ("angry", "frustrated") and resolution_attempts >= 2:
            return "high"
        if resolution_attempts >= 3:
            return "high"
        if sentiment in ("angry", "frustrated"):
            return "medium"
        if resolution_attempts >= 1:
            return "medium"
        return "low"

    def _extract_key_issue(self, state: ConversationState) -> str:
        intent = state.extracted_entities.get("intent", "unknown")
        issue_type = state.extracted_entities.get("issue_type", intent)
        for msg in reversed(state.messages):
            if msg.role == MessageRole.USER:
                return f"{issue_type}: {msg.content[:200]}"
        return str(issue_type)

    def _recommended_action(self, priority: str, sentiment: str) -> str:
        if priority == "critical":
            return "Immediate callback by senior support manager within 30 minutes"
        if priority == "high":
            return "Human operator response within 1 hour (Enterprise SLA) or 4 hours (Pro)"
        if sentiment in ("angry", "frustrated"):
            return "Empathetic human follow-up within 4 business hours"
        return "Standard escalation queue — response within 24 business hours"

    def _build_summary(self, state: ConversationState) -> str:
        lines: list[str] = []
        for msg in state.messages:
            prefix = msg.role.value.capitalize()
            agent = f" ({msg.agent_name})" if msg.agent_name else ""
            lines.append(f"{prefix}{agent}: {msg.content[:300]}")
        return "\n".join(lines[-15:])

    def _format_human_alert(self, package: EscalationPackage) -> str:
        return (
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║                    HUMAN OPERATOR ALERT                      ║\n"
            "╠══════════════════════════════════════════════════════════════╣\n"
            f"║ Conversation ID : {package.conversation_id[:36]:<40} ║\n"
            f"║ Customer ID     : {(package.customer_id or 'N/A')[:40]:<40} ║\n"
            f"║ Priority        : {package.priority:<40} ║\n"
            f"║ Sentiment       : {package.sentiment:<40} ║\n"
            f"║ Issue Type      : {package.issue_type[:40]:<40} ║\n"
            f"║ Key Issue       : {package.key_issue[:40]:<40} ║\n"
            "╠══════════════════════════════════════════════════════════════╣\n"
            f"║ Recommended Action:\n"
            f"║   {package.recommended_action[:60]}\n"
            "╠══════════════════════════════════════════════════════════════╣\n"
            "║ SUMMARY:\n"
            + "\n".join(
                f"║   {line[:60]}"
                for line in package.summary.split("\n")[:8]
            )
            + "\n"
            "╚══════════════════════════════════════════════════════════════╝"
        )

    def _build_escalation_package(self, state: ConversationState) -> EscalationPackage:
        sentiment = self._detect_sentiment(state)
        resolution_attempts = self._count_resolution_attempts(state)
        priority = self._assign_priority(state, sentiment, resolution_attempts)
        customer_id = state.extracted_entities.get("customer_id")
        issue_type = str(
            state.extracted_entities.get("issue_type")
            or state.extracted_entities.get("intent", "escalation")
        )

        return EscalationPackage(
            conversation_id=state.conversation_id,
            summary=self._build_summary(state),
            priority=priority,
            sentiment=sentiment,
            customer_id=str(customer_id) if customer_id else None,
            issue_type=issue_type,
            key_issue=self._extract_key_issue(state),
            recommended_action=self._recommended_action(priority, sentiment),
        )

    def run(self, conversation_state: ConversationState) -> AgentResponse:
        user_message = self._get_latest_user_message(conversation_state)

        if self.logger:
            self.logger.agent_invoked(agent_name=self.name)

        package = self._build_escalation_package(conversation_state)
        conversation_state.extracted_entities["escalation_package"] = package.model_dump()
        conversation_state.status = ConversationStatus.ESCALATED

        if self.logger:
            self.logger.escalation_triggered(
                priority=package.priority,
                issue_type=package.issue_type,
                sentiment=package.sentiment,
            )

        alert_block = self._format_human_alert(package)
        customer_message = (
            f"Thank you for your patience. I've escalated your case to a human specialist.\n\n"
            f"**Priority:** {package.priority.upper()}\n"
            f"**Expected response:** {package.recommended_action}\n\n"
            f"A member of our team will review your full conversation history and "
            f"contact you using the information on your account.\n\n"
            f"**Reference:** {conversation_state.conversation_id[:8].upper()}"
        )

        full_content = f"{customer_message}\n\n```\n{alert_block}\n```"

        return AgentResponse(
            content=full_content,
            agent_name=self.name,
            kb_sources_cited=[],
            requires_handover=False,
            confidence=1.0,
        )
