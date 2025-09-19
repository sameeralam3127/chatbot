# app.py
import streamlit as st
import requests
import json

# Optional imports for different APIs
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
except ImportError:
    anthropic = None

st.set_page_config(page_title="💬 Multi-API Chatbot", layout="wide")
st.title("💬 Multi-API Chatbot (Mac-ready)")

# ------------------------
# API Selection
# ------------------------
api_name = st.selectbox("Select API", ["ollama", "openai", "claude"])

# Inputs for API keys
api_key = None
if api_name in ["openai", "claude"]:
    api_key = st.text_input(f"{api_name} API Key", type="password")

# Ollama model selection
ollama_model = "llama3.1:8b"
if api_name == "ollama":
    ollama_model = st.text_input("Ollama Model Name", value="llama3.1:8b")

# ------------------------
# ChatAPI class
# ------------------------
class ChatAPI:
    def __init__(self, api_name, api_key=None, ollama_model="llama3.1:8b"):
        self.api_name = api_name
        self.api_key = api_key
        self.ollama_model = ollama_model

        if api_name == "openai" and OpenAI:
            self.client = OpenAI(api_key=api_key)
        elif api_name == "claude" and anthropic:
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.client = None

    def send_message(self, messages):
        # Combine user messages into a single prompt
        prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages if m['role'] == 'user'])

        # ---------------- OpenAI ----------------
        if self.api_name == "openai":
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"OpenAI Error: {e}"

        # ---------------- Ollama (using direct HTTP API) ----------------
        elif self.api_name == "ollama":
            try:
                # Use Ollama's HTTP API directly
                url = "http://localhost:11434/api/chat"
                payload = {
                    "model": self.ollama_model,
                    "messages": messages,
                    "stream": False
                }
                
                response = requests.post(url, json=payload)
                response.raise_for_status()
                
                result = response.json()
                return result.get("message", {}).get("content", "No content received")
                
            except requests.exceptions.ConnectionError:
                return "Error: Could not connect to Ollama. Make sure Ollama is running on your Mac (run 'ollama serve' in terminal)."
            except Exception as e:
                return f"Ollama Error: {e}"

        # ---------------- Claude ----------------
        elif self.api_name == "claude":
            try:
                # Convert messages to Claude format
                prompt = "\n\n".join([f"{m['role']}: {m['content']}" for m in messages])
                prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
                
                response = self.client.completions.create(
                    model="claude-2",
                    prompt=prompt,
                    max_tokens_to_sample=1000
                )
                return response.completion
            except Exception as e:
                return f"Claude Error: {e}"

        return "API not configured correctly."

# ------------------------
# Initialize API
# ------------------------
if api_name:
    chat_api = ChatAPI(api_name, api_key=api_key, ollama_model=ollama_model)

# ------------------------
# Session state for chat messages
# ------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get API response
    with st.spinner("Thinking..."):
        response = chat_api.send_message(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

# ------------------------
# Instructions for Ollama
# ------------------------
if api_name == "ollama":
    st.sidebar.info("""
    **Ollama Setup Instructions:**
    
    1. Install Ollama: Visit [ollama.ai](https://ollama.ai)
    2. Download and install for macOS
    3. Pull a model: `ollama pull llama3.1:8b`
    4. Start Ollama: `ollama serve`
    5. Make sure it's running on http://localhost:11434
    """)