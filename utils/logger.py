"""Structured JSON logging for CloudDash support system."""

import logging
import sys
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import structlog


class EventType(str, Enum):
    AGENT_INVOKED = "AGENT_INVOKED"
    KB_RETRIEVED = "KB_RETRIEVED"
    HANDOVER_INITIATED = "HANDOVER_INITIATED"
    HANDOVER_COMPLETED = "HANDOVER_COMPLETED"
    HANDOVER_FAILED = "HANDOVER_FAILED"
    ESCALATION_TRIGGERED = "ESCALATION_TRIGGERED"
    GUARDRAIL_TRIGGERED = "GUARDRAIL_TRIGGERED"
    LLM_CALL = "LLM_CALL"
    CONVERSATION_CREATED = "CONVERSATION_CREATED"
    MESSAGE_RECEIVED = "MESSAGE_RECEIVED"


def _add_standard_fields(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    event_dict.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog with JSON output and standard fields."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _add_standard_fields,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "clouddash") -> structlog.BoundLogger:
    return structlog.get_logger(name)


class SupportLogger:
    """Logger that enforces required fields on every support system event."""

    def __init__(
        self,
        trace_id: str,
        conversation_id: str,
        agent_name: Optional[str] = None,
    ):
        self.trace_id = trace_id
        self.conversation_id = conversation_id
        self.agent_name = agent_name
        self._logger = get_logger()

    def _base_context(self, event_type: EventType, **kwargs: Any) -> dict[str, Any]:
        ctx: dict[str, Any] = {
            "trace_id": self.trace_id,
            "conversation_id": self.conversation_id,
            "event_type": event_type.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if self.agent_name:
            ctx["agent_name"] = self.agent_name
        ctx.update(kwargs)
        return ctx

    def log(self, event_type: EventType, **kwargs: Any) -> None:
        self._logger.info(event_type.value, **self._base_context(event_type, **kwargs))

    def agent_invoked(self, agent_name: str, **kwargs: Any) -> None:
        self.log(EventType.AGENT_INVOKED, agent_name=agent_name, **kwargs)

    def kb_retrieved(self, query: str, sources: list[str], **kwargs: Any) -> None:
        self.log(
            EventType.KB_RETRIEVED,
            query=query,
            sources=sources,
            source_count=len(sources),
            **kwargs,
        )

    def handover_initiated(
        self,
        source_agent: str,
        target_agent: str,
        reason: str,
        **kwargs: Any,
    ) -> None:
        self.log(
            EventType.HANDOVER_INITIATED,
            source_agent=source_agent,
            target_agent=target_agent,
            reason=reason,
            **kwargs,
        )

    def handover_completed(self, source_agent: str, target_agent: str, **kwargs: Any) -> None:
        self.log(
            EventType.HANDOVER_COMPLETED,
            source_agent=source_agent,
            target_agent=target_agent,
            **kwargs,
        )

    def handover_failed(self, source_agent: str, target_agent: str, error: str, **kwargs: Any) -> None:
        self.log(
            EventType.HANDOVER_FAILED,
            source_agent=source_agent,
            target_agent=target_agent,
            error=error,
            **kwargs,
        )

    def escalation_triggered(self, priority: str, issue_type: str, **kwargs: Any) -> None:
        self.log(
            EventType.ESCALATION_TRIGGERED,
            priority=priority,
            issue_type=issue_type,
            **kwargs,
        )

    def guardrail_triggered(self, rule: str, action: str, **kwargs: Any) -> None:
        self.log(
            EventType.GUARDRAIL_TRIGGERED,
            rule=rule,
            action=action,
            **kwargs,
        )

    def llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        **kwargs: Any,
    ) -> None:
        self.log(
            EventType.LLM_CALL,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            **kwargs,
        )
