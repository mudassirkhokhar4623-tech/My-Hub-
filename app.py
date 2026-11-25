import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load the API key from a local .env file (only used on your own computer)
load_dotenv()

# The Gemini model to use. If Google retires it, change this one line.
MODEL_NAME = "gemini-2.5-flash"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, friendly assistant. Keep your answers clear and simple."
)

st.set_page_config(page_title="My Chatbot", page_icon="💬")


def get_api_key():
    """Find the API key: first in .env (local), then in Streamlit secrets (cloud)."""
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


api_key = get_api_key()
if not api_key:
    st.error(
        "API key not found. Locally, add GEMINI_API_KEY to your .env file. "
        "On Streamlit Cloud, add it under Settings > Secrets."
    )
    st.stop()

client = genai.Client(api_key=api_key)

# st.session_state keeps data alive while the user chats
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    system_prompt = st.text_area(
        "System prompt (the bot's personality)",
        value=DEFAULT_SYSTEM_PROMPT,
        height=150,
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Main chat area ----------
st.title("💬 My Chatbot")

# Show all earlier messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Wait for the user to type something
user_input = st.chat_input("Type your message here...")

if user_input:
    # Save and show the user's message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Convert our history into the format Gemini expects.
    # Gemini calls the bot "model", not "assistant".
    history = [
        types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[types.Part(text=m["content"])],
        )
        for m in st.session_state.messages
    ]

    def stream_reply():
        """Yield the answer piece by piece so it appears word by word."""
        stream = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=history,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(stream_reply())
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as error:
            st.error(f"Something went wrong: {error}")
            # Remove the unanswered message so the history stays clean
            st.session_state.messages.pop()
