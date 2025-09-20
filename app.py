# app.py
import streamlit as st
from chat_api import OllamaAPI
from rag import RAGEngine
from mcp import MCPServer
import db

st.set_page_config(page_title="Local Chatbot", layout="wide")
st.title("Local Chatbot (Ollama)")

# init DB
db.init_db()

# --- Sidebar: model selection / RAG / MCP toggles ---
ollama = OllamaAPI()
models = ollama.list_models()
if not models:
    # fallback list
    models = ["llama3.1:8b"]
model = st.sidebar.selectbox("Select Ollama Model", models)
ollama.model = model

st.sidebar.header("RAG Settings")
enable_rag = st.sidebar.checkbox("Enable RAG", value=False)
rag_engine = RAGEngine() if enable_rag else None

uploaded = st.sidebar.file_uploader("Upload documents (RAG)", accept_multiple_files=True, type=["txt","pdf","docx","md"])
if enable_rag and uploaded:
    count = rag_engine.process_files(uploaded)
    st.sidebar.success(f"Processed {count} chunks for RAG")

st.sidebar.header("MCP Settings")
enable_mcp = st.sidebar.checkbox("Enable MCP", value=False)
mcp = MCPServer() if enable_mcp else None



# --- main chat area ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role","content"}

# display history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# input
prompt = st.chat_input("Type your message...")
if prompt:
    # save user message to session & DB
    st.session_state.messages.append({"role": "user", "content": prompt})
    db.save_message("user", prompt)

    # Build a local-only context (we decided to keep LLM input pure)
    # If you want RAG/MCP to inform the assistant but not be printed, you can attach it to system messages
    # For now: keep the user's query clean when sent to Ollama.
    # Optionally compute RAG & MCP context and show in sidebar:
    if enable_rag and rag_engine:
        relevant = rag_engine.retrieve(prompt, top_k=3)
        if relevant:
            st.sidebar.subheader("RAG Matches")
            for r in relevant:
                st.sidebar.markdown(f"**{r['filename']}** — score {r['similarity']:.2f}")
                st.sidebar.text(r['snippet'])

    if enable_mcp and mcp:
        st.sidebar.subheader("MCP Context")
        st.sidebar.text(mcp.gather_context(prompt))

    # Prepare messages for sending: only keep user & assistant messages
    # (chat_api will also clean them)
    send_messages = st.session_state.messages.copy()

    # Send to Ollama and stream response
    with st.chat_message("assistant"):
        response_area = st.empty()
        accumulated = ""
        for chunk in ollama.chat(send_messages, stream=True):
            accumulated += chunk
            response_area.markdown(accumulated)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": accumulated})
    db.save_message("assistant", accumulated)
