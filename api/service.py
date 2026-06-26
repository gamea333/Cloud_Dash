"""Conversation orchestration service."""

import os
from typing import Optional

from dotenv import load_dotenv

from agents.orchestrator import Orchestrator
from models import AgentResponse, ConversationState, Message
from retrieval.retriever import KnowledgeRetriever
from utils.groq_client import GroqClient
from utils.logger import EventType, SupportLogger, configure_logging

load_dotenv()


class ConversationService:
    """Manages conversation state and delegates routing to the orchestrator."""

    def __init__(self) -> None:
        log_level = os.getenv("LOG_LEVEL", "INFO")
        configure_logging(log_level)

        self._groq: Optional[GroqClient] = None
        self._retriever: Optional[KnowledgeRetriever] = None
        self._orchestrator: Optional[Orchestrator] = None
        self.conversations: dict[str, ConversationState] = {}

    @property
    def groq(self) -> GroqClient:
        if self._groq is None:
            self._groq = GroqClient()
        return self._groq

    @property
    def retriever(self) -> KnowledgeRetriever:
        if self._retriever is None:
            self._retriever = KnowledgeRetriever(groq_client=self.groq)
            self._retriever.ensure_indexed()
        return self._retriever

    @property
    def orchestrator(self) -> Orchestrator:
        if self._orchestrator is None:
            self._orchestrator = Orchestrator.create(
                groq_client=self.groq,
                retriever=self.retriever,
            )
        return self._orchestrator

    def create_conversation(self) -> ConversationState:
        state = ConversationState()
        self.conversations[state.conversation_id] = state
        logger = SupportLogger(
            trace_id=state.trace_id,
            conversation_id=state.conversation_id,
        )
        logger.log(EventType.CONVERSATION_CREATED)
        return state

    def get_conversation(self, conversation_id: str) -> Optional[ConversationState]:
        return self.conversations.get(conversation_id)

    def process_message(self, conversation_id: str, content: str) -> AgentResponse:
        state = self.conversations.get(conversation_id)
        if state is None:
            raise KeyError(f"Conversation {conversation_id} not found")

        return self.orchestrator.route(state, content)

    def get_history(self, conversation_id: str) -> list[Message]:
        state = self.conversations.get(conversation_id)
        if state is None:
            raise KeyError(f"Conversation {conversation_id} not found")
        return state.messages

    def get_handover_logs(self, conversation_id: str) -> list:
        if conversation_id not in self.conversations:
            raise KeyError(f"Conversation {conversation_id} not found")
        return self.orchestrator.get_handover_log(conversation_id)
