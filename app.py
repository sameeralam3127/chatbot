import streamlit as st
from chat_api import OllamaAPI
from rag import RAGEngine
from mcp import MCPServer

st.set_page_config(page_title="💬 Local Chatbot with Ollama + RAG + MCP", layout="wide")
st.title("💬 Local Chatbot with RAG + MCP")

# --- Ollama ---
ollama_client = OllamaAPI()
available_models = ollama_client.list_models()
selected_model = st.sidebar.selectbox("Select Ollama Model", available_models)
ollama_client.model = selected_model

# --- RAG ---
st.sidebar.header("RAG Settings")
enable_rag = st.sidebar.checkbox("Enable RAG", value=True)
rag_engine = RAGEngine()

uploaded_files = st.sidebar.file_uploader(
    "Upload documents", type=["txt", "pdf", "docx", "md"], accept_multiple_files=True
)

if enable_rag and uploaded_files:
    rag_engine.process_files(uploaded_files)
    st.sidebar.success(f"Processed {len(rag_engine.documents)} chunks for RAG")

# --- MCP ---
st.sidebar.header("MCP Settings")
enable_mcp = st.sidebar.checkbox("Enable MCP", value=True)
mcp_server = MCPServer()

# --- Session state ---
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

    # --- Build context ---
    context = ""
    if enable_rag and rag_engine.documents:
        relevant = rag_engine.retrieve(prompt)
        if relevant:
            context += "Relevant info from your files:\n"
            for doc in relevant:
                context += f"- {doc['filename']}: {doc['snippet']}\n"

    if enable_mcp:
        context += "\nMCP resources:\n"
        context += mcp_server.gather_context(prompt)

    # Inject context
    final_prompt = context + "\n\n" + prompt
    st.session_state.messages[-1]["content"] = final_prompt

    with st.chat_message("assistant"):
        response_area = st.empty()
        collected = ""
        for chunk in ollama_client.chat(st.session_state.messages, stream=True):
            collected += chunk
            response_area.markdown(collected)
        st.session_state.messages.append({"role": "assistant", "content": collected})
