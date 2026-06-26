"""Technical support agent — KB-grounded troubleshooting."""

import re
from typing import Any

from models import AgentResponse, ConversationState, MessageRole
from agents.base import BaseAgent

SIMILARITY_THRESHOLD = 0.45
HELPFUL_FOOTER = (
    "\n\n---\n"
    "**Was this helpful?** Reply `escalate` if you need human support."
)

TECHNICAL_SYSTEM = """You are the CloudDash Technical Support agent. Provide accurate, step-by-step
troubleshooting using ONLY the knowledge base context provided.

Rules:
1. Use numbered steps (1., 2., 3.) for troubleshooting procedures
2. Include code snippets or config examples where relevant (API calls, IAM policies, curl commands)
3. Do NOT invent features, pricing, or procedures not in the knowledge base context
4. Be concise but thorough — assume the customer is a DevOps/SRE engineer
5. If KB context is insufficient, say so clearly

Knowledge base context:
{kb_context}
"""


class TechnicalSupportAgent(BaseAgent):
    """Handles technical, integration, and account-access troubleshooting."""

    name = "TechnicalSupport"

    def _filter_relevant_chunks(
        self, chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        relevant: list[dict[str, Any]] = []
        for chunk in chunks:
            distance = chunk.get("distance")
            if distance is None:
                relevant.append(chunk)
                continue
            similarity = 1.0 - float(distance)
            if similarity >= SIMILARITY_THRESHOLD:
                relevant.append(chunk)
        return relevant

    def _detect_billing_intent(self, text: str) -> bool:
        billing_signals = [
            "invoice", "refund", "payment", "billing", "charge",
            "subscription", "plan upgrade", "plan downgrade", "pricing",
        ]
        text_lower = text.lower()
        return any(signal in text_lower for signal in billing_signals)

    def _no_kb_response(self, citations: str) -> str:
        body = (
            "I searched our knowledge base but couldn't find a verified article "
            "that matches your issue closely enough to provide accurate guidance.\n\n"
            "I don't want to speculate about CloudDash configuration or features. "
            "I can escalate this to a human specialist who can investigate directly.\n\n"
            "Reply **escalate** to connect with our Escalation team."
        )
        if citations:
            body += f"\n\n**Sources consulted (low relevance):**\n{citations}"
        return body + HELPFUL_FOOTER

    def run(self, conversation_state: ConversationState) -> AgentResponse:
        user_message = self._get_latest_user_message(conversation_state)

        if self.logger:
            self.logger.agent_invoked(agent_name=self.name)

        if self._detect_billing_intent(user_message):
            if "Billing" in self.can_handover_to:
                return AgentResponse(
                    content=(
                        "This looks like a billing-related question. "
                        "I'm transferring you to our Billing team."
                    ),
                    agent_name=self.name,
                    requires_handover=True,
                    handover_target="Billing",
                    confidence=0.9,
                )

        chunks, sources, citations = self._retrieve_and_cite(
            user_message, conversation_state
        )
        relevant_chunks = self._filter_relevant_chunks(chunks)

        if not relevant_chunks:
            return AgentResponse(
                content=self._no_kb_response(citations),
                agent_name=self.name,
                kb_sources_cited=sources,
                requires_handover=False,
                confidence=0.4,
            )

        kb_context = "\n\n".join(c["content"] for c in relevant_chunks)
        system = TECHNICAL_SYSTEM.format(kb_context=kb_context)
        context = self._build_context(conversation_state)

        content = self._call_llm(system, user_message, context)

        if not re.search(r"^\s*1[\.\)]", content, re.MULTILINE):
            content = self._ensure_numbered_steps(content)

        if citations:
            content += f"\n\n**Sources:**\n{citations}"
        content += HELPFUL_FOOTER

        escalation_requested = "escalate" in user_message.lower()
        return AgentResponse(
            content=content,
            agent_name=self.name,
            kb_sources_cited=list({c["article_id"] for c in relevant_chunks}),
            requires_handover=escalation_requested,
            handover_target="Escalation" if escalation_requested else None,
            confidence=0.85,
        )

    def _ensure_numbered_steps(self, content: str) -> str:
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        if len(lines) <= 1:
            return f"1. {content}"
        numbered: list[str] = []
        for idx, line in enumerate(lines, start=1):
            if re.match(r"^\d+[\.\)]", line):
                numbered.append(line)
            else:
                numbered.append(f"{idx}. {line}")
        return "\n".join(numbered)
