"""FastAPI application for CloudDash multi-agent customer support."""

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.service import ConversationService
from models import AgentResponse, HandoverLog, Message

app = FastAPI(
    title="CloudDash Support API",
    description="Multi-agent customer support system for CloudDash cloud infrastructure monitoring",
    version="1.0.0",
)

_service: ConversationService | None = None


def get_service() -> ConversationService:
    global _service
    if _service is None:
        _service = ConversationService()
    return _service


class CreateConversationResponse(BaseModel):
    conversation_id: str
    trace_id: str


class SendMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)


class HistoryResponse(BaseModel):
    conversation_id: str
    messages: list[Message]


class HandoverLogResponse(BaseModel):
    conversation_id: str
    handovers: list[HandoverLog]


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "clouddash-support"}


@app.post("/conversations", response_model=CreateConversationResponse)
def create_conversation() -> CreateConversationResponse:
    state = get_service().create_conversation()
    return CreateConversationResponse(
        conversation_id=state.conversation_id,
        trace_id=state.trace_id,
    )


@app.post("/conversations/{conversation_id}/messages", response_model=AgentResponse)
def send_message(conversation_id: str, request: SendMessageRequest) -> AgentResponse:
    try:
        return get_service().process_message(conversation_id, request.content)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")


@app.get("/conversations/{conversation_id}/history", response_model=HistoryResponse)
def get_history(conversation_id: str) -> HistoryResponse:
    try:
        messages = get_service().get_history(conversation_id)
        return HistoryResponse(conversation_id=conversation_id, messages=messages)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")


@app.get("/conversations/{conversation_id}/handover-log", response_model=HandoverLogResponse)
def get_handover_log(conversation_id: str) -> HandoverLogResponse:
    try:
        logs = get_service().get_handover_logs(conversation_id)
        return HandoverLogResponse(conversation_id=conversation_id, handovers=logs)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")
