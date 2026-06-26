"""Pydantic models for the CloudDash multi-agent customer support system."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationStatus(str, Enum):
    ACTIVE = "active"
    HANDOVER = "handover"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"


class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=utc_now)
    agent_name: Optional[str] = None


class ConversationState(BaseModel):
    conversation_id: str = Field(default_factory=lambda: str(uuid4()))
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    messages: list[Message] = Field(default_factory=list)
    current_agent: str = "Triage"
    extracted_entities: dict[str, Any] = Field(default_factory=dict)
    status: ConversationStatus = ConversationStatus.ACTIVE


class AgentResponse(BaseModel):
    content: str
    agent_name: str
    kb_sources_cited: list[str] = Field(default_factory=list)
    requires_handover: bool = False
    handover_target: Optional[str] = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class HandoverPayload(BaseModel):
    source_agent: str
    target_agent: str
    reason: str
    conversation_summary: str
    extracted_entities: dict[str, Any] = Field(default_factory=dict)
    priority: str = "normal"
    timestamp: datetime = Field(default_factory=utc_now)


class HandoverLog(BaseModel):
    trace_id: str
    handover_payload: HandoverPayload
    context_snapshot: dict[str, Any] = Field(default_factory=dict)


class TriageResult(BaseModel):
    intent: str
    entities: dict[str, Any] = Field(default_factory=dict)
    target_agent: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class GuardrailInputResult(BaseModel):
    allowed: bool
    reason: str = ""


class GuardrailOutputResult(BaseModel):
    allowed: bool
    flagged_reason: str = ""
    sanitized_response: str = ""


class EscalationPackage(BaseModel):
    conversation_id: str
    summary: str
    priority: str
    sentiment: str
    customer_id: Optional[str] = None
    issue_type: str
    key_issue: str = ""
    recommended_action: str = ""
