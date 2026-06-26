"""Billing support agent — account lookup, invoices, and policy-backed responses."""

import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from models import AgentResponse, ConversationState, HandoverPayload, MessageRole
from agents.base import BaseAgent

REFUND_AUTHORITY_LIMIT = 500.0

BILLING_SYSTEM = """You are the CloudDash Billing Support agent. Use the knowledge base context
and account data provided to answer billing questions accurately.

Rules:
1. Always cite billing policy from the knowledge base when explaining charges or refunds
2. Reference specific invoice line items from the account data when available
3. For plan changes, explain what will happen (prorated charges, effective date)
4. Be empathetic — billing issues are stressful
5. Do NOT promise refunds outside stated KB policy

Knowledge base context:
{kb_context}

Account data:
{account_data}
"""


class BillingAgent(BaseAgent):
    """Handles subscriptions, invoices, payments, and plan changes."""

    name = "Billing"

    def _mock_account_lookup(self, customer_id: str) -> dict[str, Any]:
        """Generate deterministic realistic fake account data from customer_id."""
        seed = int(hashlib.md5(customer_id.encode()).hexdigest()[:8], 16)
        plans = ["Starter", "Pro", "Enterprise"]
        plan = plans[seed % 3]
        plan_prices = {"Starter": 49.0, "Pro": 199.0, "Enterprise": 999.0}
        amount = plan_prices[plan]

        billing_day = (seed % 28) + 1
        now = datetime.now(timezone.utc)
        billing_date = now.replace(day=min(billing_day, 28))

        if billing_date > now:
            billing_date = billing_date - timedelta(days=30)

        last_invoice = {
            "invoice_id": f"INV-2026-{seed % 10000:04d}",
            "date": (billing_date - timedelta(days=30)).strftime("%Y-%m-%d"),
            "amount": amount,
            "status": "paid" if seed % 5 != 0 else "failed",
            "line_items": [
                {"description": f"{plan} Plan — Monthly", "amount": amount},
            ],
        }
        if plan == "Pro" and seed % 3 == 0:
            last_invoice["line_items"].append(
                {"description": "Additional Team Seat (x2)", "amount": 50.0}
            )
            last_invoice["amount"] += 50.0

        return {
            "customer_id": customer_id,
            "plan": plan,
            "billing_date": billing_date.strftime("%Y-%m-%d"),
            "monthly_amount": amount,
            "payment_method": f"Visa ending {seed % 10000:04d}",
            "last_invoice": last_invoice,
            "account_status": "active" if last_invoice["status"] == "paid" else "payment_failed",
        }

    def _resolve_customer_id(self, state: ConversationState) -> str:
        customer_id = state.extracted_entities.get("customer_id")
        if customer_id:
            return str(customer_id)
        match = re.search(r"(?:cust[-_]?|customer[-_]?)((?:\w+-?){2,5})", 
                          self._get_latest_user_message(state), re.I)
        if match:
            return f"cust-{match.group(1)}"
        return f"cust-{state.conversation_id[:8]}"

    def _analyze_sentiment(self, state: ConversationState) -> str:
        user_messages = [
            m.content.lower()
            for m in state.messages[-3:]
            if m.role == MessageRole.USER
        ]
        combined = " ".join(user_messages)
        angry_words = ["furious", "angry", "lawyer", "sue", "terrible", "worst", "hate"]
        frustrated_words = ["frustrated", "disappointed", "unacceptable", "ridiculous", "again"]
        if any(w in combined for w in angry_words):
            return "angry"
        if any(w in combined for w in frustrated_words):
            return "frustrated"
        positive_words = ["thanks", "thank you", "great", "helpful", "appreciate"]
        if any(w in combined for w in positive_words):
            return "positive"
        return "neutral"

    def _needs_escalation(self, user_message: str, account: dict[str, Any]) -> tuple[bool, str]:
        text_lower = user_message.lower()
        manager_signals = [
            "speak to manager", "speak to a manager", "talk to manager",
            "talk to a manager", "supervisor", "escalate",
        ]
        if any(s in text_lower for s in manager_signals):
            return True, "Customer requested manager/supervisor"

        refund_match = re.search(r"\$?\s*(\d+(?:\.\d{2})?)\s*(?:refund|credit)", text_lower)
        if refund_match:
            amount = float(refund_match.group(1))
            if amount > REFUND_AUTHORITY_LIMIT:
                return True, f"Refund request ${amount:.2f} exceeds authority limit ${REFUND_AUTHORITY_LIMIT:.2f}"

        if "refund" in text_lower and account["last_invoice"]["amount"] > REFUND_AUTHORITY_LIMIT:
            return True, (
                f"Invoice amount ${account['last_invoice']['amount']:.2f} "
                f"exceeds refund authority"
            )

        return False, ""

    def _build_escalation_handover_payload(
        self,
        state: ConversationState,
        account: dict[str, Any],
        reason: str,
    ) -> HandoverPayload:
        sentiment = self._analyze_sentiment(state)
        return HandoverPayload(
            source_agent=self.name,
            target_agent="Escalation",
            reason=reason,
            conversation_summary=self._build_context(state, limit=10),
            extracted_entities={
                **state.extracted_entities,
                "customer_id": account["customer_id"],
                "invoice_details": account["last_invoice"],
                "account_plan": account["plan"],
                "sentiment": sentiment,
            },
            priority="high",
        )

    def _detect_plan_change(self, text: str) -> str | None:
        text_lower = text.lower()
        if "upgrade" in text_lower:
            return "upgrade"
        if "downgrade" in text_lower:
            return "downgrade"
        return None

    def _simulate_plan_change(self, account: dict[str, Any], change: str) -> str:
        current = account["plan"]
        tiers = ["Starter", "Pro", "Enterprise"]
        idx = tiers.index(current)
        if change == "upgrade" and idx < len(tiers) - 1:
            new_plan = tiers[idx + 1]
        elif change == "downgrade" and idx > 0:
            new_plan = tiers[idx - 1]
        else:
            return f"Your account is already on the {current} plan — no {change} is available."

        prices = {"Starter": 49.0, "Pro": 199.0, "Enterprise": 999.0}
        effective = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
        return (
            f"**Simulated Plan {change.title()}:** {current} → {new_plan}\n"
            f"- New monthly rate: ${prices[new_plan]:.2f}/month\n"
            f"- Effective date: {effective}\n"
            f"- Prorated charge for remainder of billing cycle will appear on next invoice\n"
            f"_(This is a simulated change for demonstration purposes.)_"
        )

    def run(self, conversation_state: ConversationState) -> AgentResponse:
        user_message = self._get_latest_user_message(conversation_state)

        if self.logger:
            self.logger.agent_invoked(agent_name=self.name)

        customer_id = self._resolve_customer_id(conversation_state)
        account = self._mock_account_lookup(customer_id)
        conversation_state.extracted_entities["customer_id"] = customer_id
        conversation_state.extracted_entities["plan_type"] = account["plan"]
        conversation_state.extracted_entities["invoice_details"] = account["last_invoice"]

        needs_escalation, escalation_reason = self._needs_escalation(user_message, account)
        if needs_escalation:
            payload = self._build_escalation_handover_payload(
                conversation_state, account, escalation_reason
            )
            conversation_state.extracted_entities["pending_handover"] = payload.model_dump()
            return AgentResponse(
                content=(
                    f"I understand this requires senior review. "
                    f"I'm escalating your case to our Escalation team.\n\n"
                    f"**Reason:** {escalation_reason}\n"
                    f"**Account:** {customer_id} ({account['plan']} plan)\n"
                    f"**Last invoice:** {account['last_invoice']['invoice_id']} — "
                    f"${account['last_invoice']['amount']:.2f}"
                ),
                agent_name=self.name,
                requires_handover=True,
                handover_target="Escalation",
                confidence=0.95,
            )

        chunks, sources, citations = self._retrieve_and_cite(
            user_message, conversation_state
        )
        kb_context = "\n\n".join(c["content"] for c in chunks) if chunks else "No KB articles found."
        account_str = (
            f"Customer ID: {account['customer_id']}\n"
            f"Plan: {account['plan']} (${account['monthly_amount']:.2f}/mo)\n"
            f"Billing date: {account['billing_date']}\n"
            f"Payment method: {account['payment_method']}\n"
            f"Account status: {account['account_status']}\n"
            f"Last invoice: {account['last_invoice']}"
        )

        plan_change = self._detect_plan_change(user_message)
        plan_change_note = ""
        if plan_change:
            plan_change_note = self._simulate_plan_change(account, plan_change)

        system = BILLING_SYSTEM.format(kb_context=kb_context, account_data=account_str)
        context = self._build_context(conversation_state)
        content = self._call_llm(system, user_message, context)

        if plan_change_note:
            content = f"{plan_change_note}\n\n{content}"

        if citations:
            content += f"\n\n**Policy sources:**\n{citations}"

        if account["account_status"] == "payment_failed":
            content += (
                "\n\n**Payment failure detected** on your account. "
                "Please update your payment method under Settings > Billing > Payment Methods."
            )

        return AgentResponse(
            content=content,
            agent_name=self.name,
            kb_sources_cited=sources,
            requires_handover=False,
            confidence=0.88,
        )
