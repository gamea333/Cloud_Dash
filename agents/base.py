"""Abstract base class for CloudDash support agents."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from models import AgentResponse, ConversationState, Message, MessageRole
from retrieval.retriever import KnowledgeRetriever
from utils.groq_client import DEFAULT_MODEL, GroqClient
from utils.logger import SupportLogger


class BaseAgent(ABC):
    """Abstract base for all CloudDash support agents."""

    name: str = "BaseAgent"
    CONTEXT_MESSAGE_LIMIT: int = 10

    def __init__(
        self,
        config: dict[str, Any],
        retriever: KnowledgeRetriever,
        groq_client: GroqClient,
        logger: Optional[SupportLogger] = None,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.groq = groq_client
        self.logger = logger

    def set_logger(self, logger: SupportLogger) -> None:
        self.logger = logger

    @property
    def system_prompt(self) -> str:
        return self.config.get("system_prompt", "")

    @property
    def model(self) -> str:
        return self.config.get("model", DEFAULT_MODEL)

    @property
    def temperature(self) -> float:
        return float(self.config.get("temperature", 0.3))

    @property
    def max_tokens(self) -> int:
        return int(self.config.get("max_tokens", 1024))

    @property
    def can_handover_to(self) -> list[str]:
        routing = self.config.get("routing_rules", {})
        return list(routing.get("can_handover_to", []))

    def _get_latest_user_message(self, state: ConversationState) -> str:
        for msg in reversed(state.messages):
            if msg.role == MessageRole.USER:
                return msg.content
        return ""

    def _build_context(self, state: ConversationState, limit: Optional[int] = None) -> str:
        """Format the last N messages as a context string."""
        n = limit or self.CONTEXT_MESSAGE_LIMIT
        lines: list[str] = []
        for msg in state.messages[-n:]:
            agent = f" [{msg.agent_name}]" if msg.agent_name else ""
            lines.append(f"{msg.role.value}{agent}: {msg.content}")
        return "\n".join(lines)

    def _call_llm(self, system_prompt: str, user_message: str, context: str) -> str:
        """Invoke Groq chat completion and return the assistant text."""
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if context:
            messages.append(
                {
                    "role": "system",
                    "content": f"Conversation context:\n{context}",
                }
            )
        messages.append({"role": "user", "content": user_message})
        return self.groq.chat_completion(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def _retrieve_and_cite(
        self,
        query: str,
        state: ConversationState,
        top_k: int = 3,
    ) -> tuple[list[dict[str, Any]], list[str], str]:
        """Retrieve KB chunks and build a citation block."""
        chunks = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            conversation_history=state.messages,
            rewrite=True,
        )
        sources = list({c["article_id"] for c in chunks})
        citations = self.retriever.format_citations(chunks)
        return chunks, sources, citations

    @abstractmethod
    def run(self, conversation_state: ConversationState) -> AgentResponse:
        """Execute the agent against the current conversation state."""
        ...
