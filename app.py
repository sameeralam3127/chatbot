# app.py
import streamlit as st
from chat_api import OllamaAPI
from rag import RAGEngine
from mcp import MCPServer
import db

st.set_page_config(page_title="Local Chatbot", layout="wide")
st.title("Ollama Chatbot with RAG & MCP")

# --- Init DB ---
db.init_db()

# --- Sidebar: model selection ---
ollama = OllamaAPI()
models = ollama.list_models() or ["llama3.1:8b"]
model = st.sidebar.selectbox("Select Ollama Model", models)
ollama.model = model

# --- RAG settings ---
st.sidebar.header("RAG Settings")
enable_rag = st.sidebar.checkbox("Enable RAG", value=False)
rag_engine = RAGEngine() if enable_rag else None

uploaded = st.sidebar.file_uploader(
    "Upload documents (RAG)",
    accept_multiple_files=True,
    type=["txt", "pdf", "docx", "md"]
)
if enable_rag and uploaded:
    count = rag_engine.process_files(uploaded)
    st.sidebar.success(f"Processed {count} chunks for RAG")

# --- MCP settings ---
st.sidebar.header("MCP Settings")
enable_mcp = st.sidebar.checkbox("Enable MCP", value=False)
mcp = MCPServer() if enable_mcp else None

# --- Chat state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display chat history ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- Input ---
prompt = st.chat_input("Type your message...")
if prompt:
    # 1️⃣ Store clean user message for display
    st.session_state.messages.append({"role": "user", "content": prompt})
    db.save_message("user", prompt)

    # 2️⃣ Prepare copy for LLM
    send_messages = st.session_state.messages.copy()

    rag_refs = []
    use_rag = False

    # 3️⃣ RAG retrieval
    if enable_rag and rag_engine:
        relevant = rag_engine.retrieve(prompt, top_k=3)
        if relevant:
            use_rag = True
            rag_text = "\n\n".join([f"From {r['filename']}:\n{r['snippet']}" for r in relevant])
            rag_refs = [r["filename"] for r in relevant]
            send_messages.insert(
                -1,
                {"role": "system", "content": f"Use ONLY this context to answer the question:\n\n{rag_text}"}
            )

    # 4️⃣ MCP context
    if enable_mcp and mcp:
        mcp_context = mcp.gather_context(prompt)
        if mcp_context:
            send_messages.insert(
                0,
                {"role": "system", "content": f"Extra context from MCP:\n{mcp_context}"}
            )

    # 5️⃣ Generate assistant response
    with st.chat_message("assistant"):
        response_area = st.empty()
        accumulated = ""
        for chunk in ollama.chat(send_messages, stream=True):
            accumulated += chunk
            response_area.markdown(accumulated)

    # 6️⃣ Append references only if RAG returned results
    if use_rag and rag_refs:
        accumulated += "\n\n**References:** " + ", ".join(set(rag_refs))
        response_area.markdown(accumulated)

    # 7️⃣ Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": accumulated})
    db.save_message("assistant", accumulated)
