import os

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Chat with GPT", page_icon="💬")
st.title("💬 Chat with GPT")

# Optional: allow typing the API key in the sidebar (useful on Streamlit Cloud)
with st.sidebar:
    st.subheader("Settings")
    user_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    base_url = st.text_input("Custom API Base (optional)", placeholder="")
    model = st.text_input("Model", value="gpt-4o-mini")
    st.caption("Tip: keep a short system prompt for tone/behavior.")
    system_prompt = st.text_area("System prompt", value="You are a helpful assistant.", height=80)

api_key = user_api_key or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.info("Enter your OpenAI API key in the sidebar or set OPENAI_API_KEY.", icon="🔑")
    st.stop()

client = OpenAI(api_key=api_key, base_url=base_url or None)

# Keep conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
if prompt := st.chat_input("Type your message…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream assistant reply
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""

        # Build messages list with an initial system prompt
        msgs = [{"role": "system", "content": system_prompt}] + st.session_state.messages

        # Stream from Chat Completions
        stream = client.chat.completions.create(
            model=model,
            messages=msgs,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                full_text += delta.content
                placeholder.markdown(full_text)

        # Save final assistant message
        st.session_state.messages.append({"role": "assistant", "content": full_text})
