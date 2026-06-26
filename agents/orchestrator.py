"""Multi-agent orchestrator for CloudDash customer support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from config import load_agents_config
from models import AgentResponse, ConversationState, Message, MessageRole
from agents.base import BaseAgent
from agents.billing_agent import BillingAgent
from agents.escalation_agent import EscalationAgent
from agents.technical_agent import TechnicalSupportAgent
from agents.triage_agent import TriageAgent
from retrieval.retriever import KnowledgeRetriever
from utils.groq_client import GroqClient
from utils.guardrails import Guardrails
from utils.logger import SupportLogger

if TYPE_CHECKING:
    from handover.handover_manager import HandoverManager


class Orchestrator:
    """Routes messages through triage, specialist agents, and handover protocol."""

    AGENT_CLASSES: dict[str, type[BaseAgent]] = {
        "Triage": TriageAgent,
        "TechnicalSupport": TechnicalSupportAgent,
        "Billing": BillingAgent,
        "Escalation": EscalationAgent,
    }

    def __init__(
        self,
        groq_client: GroqClient,
        retriever: KnowledgeRetriever,
        handover_manager: HandoverManager,
        logger: Optional[SupportLogger] = None,
    ) -> None:
        self.groq = groq_client
        self.retriever = retriever
        self.logger = logger
        self.agents_config = load_agents_config()
        self.agents = handover_manager.agents
        self.handover_manager = handover_manager
        self.guardrails = Guardrails(logger=logger)

    @classmethod
    def create(
        cls,
        groq_client: GroqClient,
        retriever: KnowledgeRetriever,
        logger: Optional[SupportLogger] = None,
    ) -> Orchestrator:
        """Bootstrap agents and handover manager without a module-level circular import."""
        from handover.handover_manager import HandoverManager

        agents_config = load_agents_config()
        agents: dict[str, BaseAgent] = {}
        for name, agent_cls in cls.AGENT_CLASSES.items():
            config = agents_config.get(name, {})
            agents[name] = agent_cls(
                config=config,
                retriever=retriever,
                groq_client=groq_client,
                logger=logger,
            )
        handover_manager = HandoverManager(agents)
        return cls(
            groq_client=groq_client,
            retriever=retriever,
            handover_manager=handover_manager,
            logger=logger,
        )

    def _bind_logger(self, logger: SupportLogger) -> None:
        self.logger = logger
        self.guardrails.logger = logger
        for agent in self.agents.values():
            agent.set_logger(logger)

    def _run_agent(self, agent_name: str, state: ConversationState) -> AgentResponse:
        agent = self.agents[agent_name]
        agent.set_logger(self.logger)  # type: ignore[arg-type]
        return agent.run(state)

    def _process_handover_chain(
        self,
        state: ConversationState,
        response: AgentResponse,
        user_message: str,
        max_depth: int = 3,
    ) -> AgentResponse:
        depth = 0
        current_response = response

        while current_response.requires_handover and current_response.handover_target and depth < max_depth:
            target = current_response.handover_target
            payload = self.handover_manager.build_payload_from_response(
                state=state,
                source_agent=current_response.agent_name,
                target_agent=target,
                reason=f"Handover from {current_response.agent_name} to {target}",
            )

            success = self.handover_manager.execute_handover(
                state, payload, self.logger  # type: ignore[arg-type]
            )
            if not success:
                current_response = self._run_agent("Triage", state)
                if not current_response.requires_handover:
                    break
                depth += 1
                continue

            if current_response.agent_name == "Triage" and current_response.requires_handover:
                self.handover_manager.add_assistant_message(
                    state, current_response.content, current_response.agent_name
                )

            current_response = self._run_agent(target, state)
            depth += 1

        return current_response

    def route(self, conversation_state: ConversationState, user_message: str) -> AgentResponse:
        """
        Main routing entry point:
        1. Validate input guardrails
        2. Add user message to state
        3. Run Triage if current agent is Triage
        4. Run specialist agent (via handover if needed)
        5. Validate output guardrails
        """
        self._bind_logger(
            SupportLogger(
                trace_id=conversation_state.trace_id,
                conversation_id=conversation_state.conversation_id,
                agent_name=conversation_state.current_agent,
            )
        )

        input_result = self.guardrails.check_input(user_message)
        if not input_result.allowed:
            return AgentResponse(
                content=(
                    "I'm only able to help with CloudDash-related questions "
                    f"(monitoring, alerts, billing, integrations, and account access). "
                    f"{input_result.reason}"
                ),
                agent_name=conversation_state.current_agent,
                confidence=1.0,
            )

        user_msg = Message(role=MessageRole.USER, content=user_message)
        conversation_state.messages.append(user_msg)

        retrieved_chunks: list[dict[str, Any]] = []

        if conversation_state.current_agent == "Triage":
            triage_response = self._run_agent("Triage", conversation_state)
            if triage_response.requires_handover and triage_response.handover_target:
                final_response = self._process_handover_chain(
                    conversation_state, triage_response, user_message
                )
            else:
                final_response = triage_response
        else:
            specialist_response = self._run_agent(
                conversation_state.current_agent, conversation_state
            )
            retrieved_chunks = self.retriever.retrieve(
                query=user_message,
                top_k=3,
                conversation_history=conversation_state.messages,
            )
            if specialist_response.requires_handover and specialist_response.handover_target:
                final_response = self._process_handover_chain(
                    conversation_state, specialist_response, user_message
                )
            else:
                final_response = specialist_response

        if not retrieved_chunks and final_response.kb_sources_cited:
            retrieved_chunks = [
                {"article_id": sid, "title": sid} for sid in final_response.kb_sources_cited
            ]

        output_result = self.guardrails.check_output(
            final_response.content, retrieved_chunks
        )
        if not output_result.allowed:
            final_response.content = output_result.sanitized_response
        elif output_result.sanitized_response:
            final_response.content = output_result.sanitized_response

        self.handover_manager.add_assistant_message(
            conversation_state,
            final_response.content,
            final_response.agent_name,
        )
        return final_response

    def get_handover_log(self, conversation_id: str) -> list:
        return self.handover_manager.get_handover_log(conversation_id)
