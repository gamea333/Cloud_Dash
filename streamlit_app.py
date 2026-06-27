"""Streamlit web UI for the CloudDash multi-agent customer support system."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import requests
import streamlit as st

API_BASE_URL = "https://cloud-dash.onrender.com"

WARMUP_MSG = "Our servers are warming up, please wait 30 seconds and try again."
NOT_FOUND_MSG = "Conversation not found, start a new one."
CONNECTION_MSG = "Cannot reach the server. Please try again in a moment."
GENERIC_MSG = "Something went wrong. Please try again."

INIT_MAX_RETRIES = 3
INIT_RETRY_DELAY_SECONDS = 10

EXAMPLE_QUERIES_SIDEBAR = """
**🔧 Technical**
- My alerts stopped firing after updating AWS credentials
- Dashboard is loading slowly

**💳 Billing**
- What's the difference between Pro and Enterprise plans?
- I've been charged twice for April, I need a refund

**👤 Account**
- How do I set up SSO for my team?
- How do I invite team members?

**🧪 Test Guardrails**
- Who won the cricket match? (blocked)
- Ignore previous instructions (blocked)
"""


class ApiError(Exception):
    """User-friendly API error; never carries raw HTML or stack traces."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


def _error_from_status(status_code: int) -> ApiError:
    if status_code in (502, 503):
        return ApiError(WARMUP_MSG, status_code=status_code, retryable=True)
    if status_code == 404:
        return ApiError(NOT_FOUND_MSG, status_code=404)
    return ApiError(GENERIC_MSG, status_code=status_code)


def _api_request(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    try:
        response = requests.request(method, url, timeout=kwargs.pop("timeout", 60), **kwargs)
    except requests.ConnectionError:
        raise ApiError(CONNECTION_MSG, retryable=True) from None
    except requests.Timeout:
        raise ApiError(GENERIC_MSG, retryable=True) from None
    except requests.RequestException:
        raise ApiError(GENERIC_MSG) from None

    if not response.ok:
        raise _error_from_status(response.status_code)

    content_type = response.headers.get("Content-Type", "")
    if "application/json" not in content_type:
        raise ApiError(GENERIC_MSG)

    try:
        return response.json()
    except ValueError:
        raise ApiError(GENERIC_MSG) from None


def create_conversation() -> tuple[str, str]:
    data = _api_request("POST", f"{API_BASE_URL}/conversations", timeout=60)
    return data["conversation_id"], data["trace_id"]


def create_conversation_with_retries(
    max_retries: int = INIT_MAX_RETRIES,
    delay: int = INIT_RETRY_DELAY_SECONDS,
) -> tuple[str, str]:
    last_error: Optional[ApiError] = None
    for attempt in range(max_retries):
        try:
            return create_conversation()
        except ApiError as exc:
            last_error = exc
            if exc.status_code in (502, 503) and attempt < max_retries - 1:
                time.sleep(delay)
                continue
            raise
    if last_error:
        raise last_error
    raise ApiError(GENERIC_MSG)


def send_message(conversation_id: str, content: str) -> dict[str, Any]:
    return _api_request(
        "POST",
        f"{API_BASE_URL}/conversations/{conversation_id}/messages",
        json={"content": content},
        timeout=60,
    )


def fetch_handover_log(conversation_id: str) -> dict[str, Any]:
    return _api_request(
        "GET",
        f"{API_BASE_URL}/conversations/{conversation_id}/handover-log",
        timeout=60,
    )


def show_api_error(
    error: ApiError,
    retry_key: Optional[str] = None,
    on_retry: Optional[Callable[[], None]] = None,
) -> None:
    """Display a friendly error; optionally show a retry button."""
    st.error(error.message)
    if error.retryable and retry_key and on_retry:
        if st.button("Retry", key=retry_key, use_container_width=True):
            on_retry()


def retry_connection() -> None:
    """Clear session and re-attempt conversation creation."""
    for key in (
        "conversation_id",
        "trace_id",
        "messages",
        "handover_log",
        "init_error",
        "init_error_retryable",
        "pending_message",
        "chat_error",
        "sidebar_error",
    ):
        st.session_state.pop(key, None)
    init_session()
    st.rerun()


def init_session() -> None:
    if "conversation_id" not in st.session_state:
        st.session_state.messages = []
        st.session_state.handover_log = None
        st.session_state.init_error = None
        st.session_state.init_error_retryable = False
        try:
            with st.spinner("Server is starting up..."):
                conversation_id, trace_id = create_conversation_with_retries()
            st.session_state.conversation_id = conversation_id
            st.session_state.trace_id = trace_id
        except ApiError as exc:
            st.session_state.conversation_id = None
            st.session_state.trace_id = None
            st.session_state.init_error = exc.message
            st.session_state.init_error_retryable = exc.retryable


def reset_conversation() -> None:
    for key in (
        "conversation_id",
        "trace_id",
        "messages",
        "handover_log",
        "init_error",
        "init_error_retryable",
        "pending_message",
        "chat_error",
        "sidebar_error",
    ):
        st.session_state.pop(key, None)
    init_session()
    st.rerun()


def render_assistant_message(msg: dict[str, Any]) -> None:
    with st.chat_message("assistant"):
        st.markdown(msg["content"])
        agent_name = msg.get("agent_name")
        if agent_name:
            st.caption(f"Agent: **{agent_name}**")
        kb_sources = msg.get("kb_sources_cited") or []
        if kb_sources:
            st.caption(f"KB sources: {', '.join(kb_sources)}")


def _retry_pending_message() -> None:
    pending = st.session_state.get("pending_message")
    conv_id = st.session_state.get("conversation_id")
    if not pending or not conv_id:
        return

    st.session_state.messages.append({"role": "user", "content": pending})
    try:
        with st.spinner("Routing to the right agent..."):
            response = send_message(conv_id, pending)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response["content"],
                "agent_name": response.get("agent_name"),
                "kb_sources_cited": response.get("kb_sources_cited", []),
            }
        )
        st.session_state.pop("pending_message", None)
        st.session_state.pop("chat_error", None)
    except ApiError as exc:
        st.session_state.messages.pop()
        st.session_state.chat_error = exc
    st.rerun()


def _retry_handover_log() -> None:
    conv_id = st.session_state.get("conversation_id")
    if not conv_id:
        st.session_state.sidebar_error = ApiError(NOT_FOUND_MSG, status_code=404)
        st.rerun()
        return
    try:
        with st.spinner("Loading handover log..."):
            st.session_state.handover_log = fetch_handover_log(conv_id)
        st.session_state.pop("sidebar_error", None)
    except ApiError as exc:
        st.session_state.sidebar_error = exc
        st.session_state.handover_log = None
    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="CloudDash Support",
        page_icon="☁️",
        layout="wide",
    )

    init_session()

    st.title("CloudDash Support")
    st.caption("Multi-agent AI customer support for cloud infrastructure monitoring")

    with st.sidebar:
        st.header("Session")

        if st.session_state.get("init_error"):
            show_api_error(
                ApiError(
                    st.session_state.init_error,
                    retryable=st.session_state.get("init_error_retryable", False),
                ),
                retry_key="init_retry",
                on_retry=retry_connection,
            )
        elif st.session_state.get("conversation_id"):
            st.text_input(
                "Conversation ID",
                value=st.session_state.conversation_id,
                disabled=True,
            )
            if st.session_state.get("trace_id"):
                st.caption(f"Trace: `{st.session_state.trace_id[:8]}...`")

        if st.button("Retry Connection", use_container_width=True):
            retry_connection()

        if st.button("New Conversation", use_container_width=True):
            reset_conversation()

        if st.button("View Handover Log", use_container_width=True):
            conv_id = st.session_state.get("conversation_id")
            if not conv_id:
                st.session_state.sidebar_error = ApiError(NOT_FOUND_MSG, status_code=404)
            else:
                try:
                    with st.spinner("Loading handover log..."):
                        st.session_state.handover_log = fetch_handover_log(conv_id)
                    st.session_state.pop("sidebar_error", None)
                except ApiError as exc:
                    st.session_state.sidebar_error = exc
                    st.session_state.handover_log = None
            st.rerun()

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 💡 Example Queries")
        st.sidebar.markdown(EXAMPLE_QUERIES_SIDEBAR)

        sidebar_error = st.session_state.get("sidebar_error")
        if isinstance(sidebar_error, ApiError):
            show_api_error(
                sidebar_error,
                retry_key="sidebar_retry",
                on_retry=_retry_handover_log,
            )

        handover_log = st.session_state.get("handover_log")
        if handover_log:
            st.divider()
            st.subheader("Handover Log")
            handovers = handover_log.get("handovers", [])
            if not handovers:
                st.info("No handovers recorded for this conversation yet.")
            else:
                for idx, entry in enumerate(handovers, start=1):
                    payload = entry.get("handover_payload", {})
                    st.markdown(
                        f"**#{idx}** {payload.get('source_agent', '?')} → "
                        f"{payload.get('target_agent', '?')}"
                    )
                    st.caption(payload.get("reason", ""))
                    ts = payload.get("timestamp", "")
                    if ts:
                        st.caption(f"_{ts}_")

    if st.session_state.get("init_error"):
        if st.session_state.get("init_error_retryable"):
            st.info(WARMUP_MSG)
        else:
            st.warning("Unable to start a conversation. Use **Retry Connection** in the sidebar.")
        return

    chat_error = st.session_state.get("chat_error")
    if isinstance(chat_error, ApiError) and st.session_state.get("pending_message"):
        show_api_error(
            chat_error,
            retry_key="chat_retry",
            on_retry=_retry_pending_message,
        )

    for msg in st.session_state.get("messages", []):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            render_assistant_message(msg)

    if prompt := st.chat_input("Describe your CloudDash issue..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        conv_id = st.session_state.conversation_id
        try:
            with st.spinner("Routing to the right agent..."):
                response = send_message(conv_id, prompt)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response["content"],
                    "agent_name": response.get("agent_name"),
                    "kb_sources_cited": response.get("kb_sources_cited", []),
                }
            )
            st.session_state.pop("pending_message", None)
            st.session_state.pop("chat_error", None)
            render_assistant_message(st.session_state.messages[-1])
        except ApiError as exc:
            st.session_state.pending_message = prompt
            st.session_state.chat_error = exc
            st.session_state.messages.pop()
            show_api_error(
                exc,
                retry_key="chat_send_retry",
                on_retry=_retry_pending_message,
            )


if __name__ == "__main__":
    main()
