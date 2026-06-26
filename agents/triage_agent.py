"""Triage agent — classifies intent and routes to specialist agents."""

from typing import Any

from config import load_routing_rules
from models import AgentResponse, ConversationState, TriageResult
from agents.base import BaseAgent
from utils.groq_client import FAST_MODEL

VALID_INTENTS = ["technical", "billing", "account", "general", "escalation", "unknown"]
CONFIDENCE_ESCALATION_THRESHOLD = 0.6

TRIAGE_JSON_PROMPT = """Analyze the customer message and classify it for CloudDash support routing.

Return JSON with exactly these fields:
- intent: one of technical, billing, account, general, escalation, unknown
- entities: object with optional keys customer_id, plan_type, product_area, urgency
- target_agent: one of Triage, TechnicalSupport, Billing, Escalation
- confidence: float 0.0-1.0
- reasoning: brief explanation of classification

Routing rules:
- technical/account issues -> TechnicalSupport
- billing issues -> Billing
- escalation requests (manager, lawyer, urgent complaint) -> Escalation
- general greetings/overview -> Triage (but will route to TechnicalSupport for product questions)
- unknown with low confidence -> Escalation

Customer message: {message}

Conversation context:
{context}
"""


class TriageAgent(BaseAgent):
    """Classifies intent and routes — never answers customer questions directly."""

    name = "Triage"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._routing = load_routing_rules()

    def _route_intent_to_agent(self, intent: str) -> str:
        intent_map: dict[str, str] = self._routing.get("intents", {})
        return intent_map.get(intent, "TechnicalSupport")

    def _classify(self, state: ConversationState, user_message: str) -> TriageResult:
        context = self._build_context(state, limit=5)
        prompt = TRIAGE_JSON_PROMPT.format(message=user_message, context=context)

        raw = self.groq.chat_completion_json(
            messages=[{"role": "user", "content": prompt}],
            model=FAST_MODEL,
            temperature=0.1,
            max_tokens=512,
        )

        intent = str(raw.get("intent", "unknown"))
        if intent not in VALID_INTENTS:
            intent = "unknown"

        entities = raw.get("entities", {})
        if not isinstance(entities, dict):
            entities = {}

        target_agent = str(raw.get("target_agent", self._route_intent_to_agent(intent)))
        if target_agent not in ("Triage", "TechnicalSupport", "Billing", "Escalation"):
            target_agent = self._route_intent_to_agent(intent)

        confidence = float(raw.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        reasoning = str(raw.get("reasoning", "Classification completed"))

        if confidence < CONFIDENCE_ESCALATION_THRESHOLD:
            target_agent = "Escalation"
            reasoning = (
                f"{reasoning} Confidence {confidence:.2f} below threshold "
                f"{CONFIDENCE_ESCALATION_THRESHOLD}; routing to Escalation."
            )

        if intent == "general" and confidence >= CONFIDENCE_ESCALATION_THRESHOLD:
            target_agent = "TechnicalSupport"

        return TriageResult(
            intent=intent,
            entities=entities,
            target_agent=target_agent,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _routing_message(self, result: TriageResult) -> str:
        agent_labels = {
            "TechnicalSupport": "Technical Support",
            "Billing": "Billing Support",
            "Escalation": "Escalation Team",
            "Triage": "our support team",
        }
        label = agent_labels.get(result.target_agent, result.target_agent)
        return (
            f"I've reviewed your request and classified it as a **{result.intent}** inquiry. "
            f"I'm connecting you with {label} who can best assist you.\n\n"
            f"_(Routing confidence: {result.confidence:.0%})_"
        )

    def run(self, conversation_state: ConversationState) -> AgentResponse:
        user_message = self._get_latest_user_message(conversation_state)
        result = self._classify(conversation_state, user_message)

        conversation_state.extracted_entities.update(result.entities)
        conversation_state.extracted_entities["intent"] = result.intent
        conversation_state.extracted_entities["triage_reasoning"] = result.reasoning

        if self.logger:
            self.logger.agent_invoked(
                agent_name=self.name,
                intent=result.intent,
                target_agent=result.target_agent,
                confidence=result.confidence,
                reasoning=result.reasoning,
            )

        requires_handover = result.target_agent != "Triage"
        return AgentResponse(
            content=self._routing_message(result),
            agent_name=self.name,
            kb_sources_cited=[],
            requires_handover=requires_handover,
            handover_target=result.target_agent if requires_handover else None,
            confidence=result.confidence,
        )
