"""Streamlit web UI for the CloudDash multi-agent customer support system."""

from __future__ import annotations

import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"


def create_conversation() -> tuple[str, str]:
    response = requests.post(f"{API_BASE_URL}/conversations", timeout=30)
    response.raise_for_status()
    data = response.json()
    return data["conversation_id"], data["trace_id"]


def send_message(conversation_id: str, content: str) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/conversations/{conversation_id}/messages",
        json={"content": content},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def fetch_handover_log(conversation_id: str) -> dict:
    response = requests.get(
        f"{API_BASE_URL}/conversations/{conversation_id}/handover-log",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def init_session() -> None:
    if "conversation_id" not in st.session_state:
        try:
            conversation_id, trace_id = create_conversation()
            st.session_state.conversation_id = conversation_id
            st.session_state.trace_id = trace_id
            st.session_state.messages = []
            st.session_state.handover_log = None
        except requests.RequestException as exc:
            st.session_state.conversation_id = None
            st.session_state.trace_id = None
            st.session_state.messages = []
            st.session_state.handover_log = None
            st.session_state.init_error = str(exc)


def reset_conversation() -> None:
    for key in ("conversation_id", "trace_id", "messages", "handover_log", "init_error"):
        st.session_state.pop(key, None)
    init_session()


def render_assistant_message(msg: dict) -> None:
    with st.chat_message("assistant"):
        st.markdown(msg["content"])
        agent_name = msg.get("agent_name")
        if agent_name:
            st.caption(f"Agent: **{agent_name}**")
        kb_sources = msg.get("kb_sources_cited") or []
        if kb_sources:
            st.caption(f"KB sources: {', '.join(kb_sources)}")


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
            st.error(
                f"Could not connect to API at {API_BASE_URL}. "
                "Make sure the FastAPI server is running."
            )
            st.caption(st.session_state.init_error)
        elif st.session_state.get("conversation_id"):
            st.text_input(
                "Conversation ID",
                value=st.session_state.conversation_id,
                disabled=True,
            )
            if st.session_state.get("trace_id"):
                st.caption(f"Trace: `{st.session_state.trace_id[:8]}...`")

        if st.button("New Conversation", use_container_width=True):
            reset_conversation()
            st.rerun()

        if st.button("View Handover Log", use_container_width=True):
            conv_id = st.session_state.get("conversation_id")
            if not conv_id:
                st.error("No active conversation.")
            else:
                try:
                    with st.spinner("Loading handover log..."):
                        st.session_state.handover_log = fetch_handover_log(conv_id)
                except requests.RequestException as exc:
                    st.error(f"Failed to load handover log: {exc}")

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
        st.warning("Start the backend with: `python -m uvicorn api.main:app --reload`")
        return

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
            assistant_msg = {
                "role": "assistant",
                "content": response["content"],
                "agent_name": response.get("agent_name"),
                "kb_sources_cited": response.get("kb_sources_cited", []),
            }
            st.session_state.messages.append(assistant_msg)
            render_assistant_message(assistant_msg)
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"API error ({exc.response.status_code if exc.response else 'unknown'}): {detail}")
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")


if __name__ == "__main__":
    main()
