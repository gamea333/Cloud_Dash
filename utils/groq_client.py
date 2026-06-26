"""Reusable Groq LLM client with retry logic and structured logging."""

import json
import os
import time
from typing import Any, Optional

from dotenv import load_dotenv 
load_dotenv()                   

from groq import Groq, RateLimitError

from utils.logger import SupportLogger

DEFAULT_MODEL = "llama-3.1-8b-instant"
FAST_MODEL = "llama-3.1-8b-instant"
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 1.0


class GroqClient:
    """Wrapper around the Groq SDK with exponential backoff and JSON parsing."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        logger: Optional[SupportLogger] = None,
    ):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        self.client = Groq(api_key=key)
        self.logger = logger

    def _log_call(
        self,
        model: str,
        usage: Any,
        latency_ms: float,
    ) -> None:
        if self.logger is None:
            return
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        self.logger.llm_call(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

    def _call_with_retry(self, **kwargs: Any) -> Any:
        model = kwargs.get("model", DEFAULT_MODEL)
        last_error: Optional[Exception] = None

        for attempt in range(MAX_RETRIES):
            try:
                start = time.perf_counter()
                response = self.client.chat.completions.create(**kwargs)
                latency_ms = (time.perf_counter() - start) * 1000
                self._log_call(model, response.usage, latency_ms)
                return response
            except RateLimitError as exc:
                last_error = exc
                if attempt < MAX_RETRIES - 1:
                    backoff = BASE_BACKOFF_SECONDS * (2 ** attempt)
                    time.sleep(backoff)
                continue

        raise last_error  # type: ignore[misc]

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        response = self._call_with_retry(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def chat_completion_json(
        self,
        messages: list[dict[str, str]],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        json_messages = messages + [
            {
                "role": "system",
                "content": (
                    "You must respond with valid JSON only. No markdown, no code fences, "
                    "no explanatory text outside the JSON object."
                ),
            }
        ]
        raw = self.chat_completion(
            messages=json_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(cleaned)
