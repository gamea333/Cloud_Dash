"""Input and output guardrails for CloudDash support."""

import re
from typing import Any, Optional

from models import GuardrailInputResult, GuardrailOutputResult
from utils.logger import SupportLogger

INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"(?i)you\s+are\s+now\s+",
    r"(?i)forget\s+your\s+",
    r"(?i)new\s+persona",
    r"(?i)\bDAN\b",
    r"(?i)jailbreak",
    r"(?i)reveal\s+(?:your\s+)?system\s+prompt",
    r"(?i)act\s+as\s+(?:if|though)\s+you\s+(?:have\s+)?no\s+restrictions",
]

CLOUDDASH_DOMAIN_KEYWORDS = [
    "cloud", "alert", "billing", "dashboard", "aws", "gcp", "azure",
    "integration", "api", "account", "plan", "invoice", "sso",
    "monitoring", "clouddash", "cloudwatch", "metric", "webhook",
    "kubernetes", "k8s", "escalate", "support", "login", "rbac",
    "refund", "payment", "subscription", "team", "uptime", "latency",
    "outage", "error", "sdk", "terraform",
]

OFF_TOPIC_PATTERNS = [
    r"(?i)\b(weather|forecast|temperature)\b",
    r"(?i)\b(football|soccer|basketball|cricket|sports|nba|nfl)\b",
    r"(?i)\b(election|politics|president|congress)\b",
    r"(?i)\b(recipe|cooking|restaurant)\b",
    r"(?i)\b(movie|netflix|celebrity)\b",
]

PII_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
    (
        r"\b(?:\d{4}[\s-]){3}\d{4}\b|"
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b",
        "credit card",
    ),
]

FEATURE_CLAIM_PATTERNS = [
    r"(?i)\$\d+(?:\.\d{2})?\s*/\s*month",
    r"(?i)(?:starter|pro|enterprise)\s+plan",
    r"(?i)cloud\s*dash\s+(?:supports|offers|provides|includes)",
    r"(?i)(?:api|endpoint)\s+(?:supports|allows|provides)",
    r"(?i)sla\s+of\s+\d+",
]

UNVERIFIED_RESPONSE = (
    "I don't have verified information on that. Let me escalate this to ensure accuracy."
)


class Guardrails:
    """Fast keyword/regex guardrails — no extra LLM calls."""

    def __init__(self, logger: Optional[SupportLogger] = None) -> None:
        self.logger = logger

    def _log_guardrail(self, rule: str, action: str, **kwargs: Any) -> None:
        if self.logger:
            self.logger.guardrail_triggered(rule=rule, action=action, **kwargs)

    def _has_domain_overlap(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in CLOUDDASH_DOMAIN_KEYWORDS)

    def _is_off_topic(self, text: str) -> bool:
        return any(re.search(pat, text) for pat in OFF_TOPIC_PATTERNS)

    def check_input(self, message: str) -> GuardrailInputResult:
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, message):
                reason = (
                    "Your message was flagged for containing instructions that "
                    "cannot be processed. Please describe your CloudDash issue directly."
                )
                self._log_guardrail("prompt_injection", "block", pattern=pattern)
                return GuardrailInputResult(allowed=False, reason=reason)

        if self._is_off_topic(message) and not self._has_domain_overlap(message):
            reason = (
                "I can only assist with CloudDash platform questions — "
                "monitoring, alerts, billing, integrations, and account access."
            )
            self._log_guardrail("off_topic", "block")
            return GuardrailInputResult(allowed=False, reason=reason)

        if not self._has_domain_overlap(message) and len(message.split()) > 8:
            if self._is_off_topic(message):
                reason = "This appears unrelated to CloudDash. How can I help with your cloud monitoring needs?"
                self._log_guardrail("off_topic_long", "block")
                return GuardrailInputResult(allowed=False, reason=reason)

        return GuardrailInputResult(allowed=True, reason="")

    def _redact_pii(self, text: str) -> str:
        redacted = text
        for pattern, label in PII_PATTERNS:
            if re.search(pattern, redacted):
                redacted = re.sub(pattern, "[REDACTED]", redacted)
                self._log_guardrail("pii_redaction", "redact", pii_type=label)
        return redacted

    def _makes_specific_claims(self, response: str) -> bool:
        return any(re.search(pat, response) for pat in FEATURE_CLAIM_PATTERNS)

    def check_output(
        self,
        response: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> GuardrailOutputResult:
        sanitized = self._redact_pii(response)

        has_relevant_kb = len(retrieved_chunks) > 0
        if not has_relevant_kb and self._makes_specific_claims(sanitized):
            self._log_guardrail(
                "unverified_claims",
                "block",
                reason="Specific claims without KB backing",
            )
            return GuardrailOutputResult(
                allowed=False,
                flagged_reason="Response contained unverified CloudDash claims without KB support",
                sanitized_response=UNVERIFIED_RESPONSE,
            )

        if sanitized != response:
            return GuardrailOutputResult(
                allowed=True,
                flagged_reason="PII redacted from response",
                sanitized_response=sanitized,
            )

        return GuardrailOutputResult(
            allowed=True,
            flagged_reason="",
            sanitized_response=sanitized,
        )
