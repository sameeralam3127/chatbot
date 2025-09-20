# Local Chatbot with RAG + MCP + SQLAlchemy

A **fast, modular, local-first chatbot** powered by [Ollama](https://ollama.ai/) models.
This project demonstrates how to combine:

- 🧠 **Chat with Ollama** (streaming for fast responses)
- 📄 **RAG** (Retrieval-Augmented Generation) with cached embeddings for local docs
- 🔌 **MCP (Model Context Protocol) simulator** for local tool access (web search, calculator, doc store)
- 🗄️ **SQLite persistence** with SQLAlchemy for storing conversations

Designed for **local use only**: no cloud dependencies, no external APIs required.

---

## 🚀 Features

- **Chatbot UI** with [Streamlit](https://streamlit.io/)
- **Ollama integration** → select any local LLM (`llama3.1:8b`, `mistral`, etc.)
- **Fast responses** via streaming chunks
- **RAG (Retrieval-Augmented Generation)**

  - Upload `.txt`, `.pdf`, `.docx`, `.md` documents
  - Chunking + embeddings with `sentence-transformers`
  - Caching for speed

- **MCP simulator**

  - Web search (stub)
  - Calculator (safe arithmetic)
  - Document store (demo reference)

- **Conversation history**

  - Stored in SQLite (`chat_history.db`)
  - SQLAlchemy ORM models for messages
  - Recent conversations shown in sidebar

---

## 📦 Installation

### 1. Clone repo

```bash
git clone https://github.com/sameeralam3127/chatbot.git
cd local-chatbot
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Ollama

```bash
ollama serve
ollama pull llama3.1:8b
```

### 5. Run Streamlit app

```bash
streamlit run app.py
```

---

## 🖥️ Usage

- Select a model from the sidebar (only chat models are shown, embedding-only models are excluded).
- Type messages in the chat input.
- (Optional) Enable **RAG** → upload documents for context retrieval.
- (Optional) Enable **MCP** → use simulated web search, calculator, or doc store.
- Messages are saved to SQLite (`chat_history.db`) automatically.

---

## 📂 Project Structure

```
local_chatbot/
│
├── app.py              # Streamlit app (UI)
├── chat_api.py         # Ollama API wrapper
├── rag.py              # RAG engine with embeddings
├── mcp.py              # MCP simulator
├── db.py               # SQLite/SQLAlchemy storage
├── requirements.txt    # Dependencies
└── README.md           # This file
```

---

## 🔮 Future Features & Enhancements

Planned improvements and ideas for extending this chatbot:

### 💡 Core Enhancements

- [ ] **Clear conversation** button → reset both session and database.
- [ ] **Conversation management** → multiple named conversations instead of one global history.
- [ ] **Better system prompts** → inject RAG/MCP context as hidden instructions, not shown to the user.
- [ ] **Support multi-turn retrieval** → context retrieved per conversation turn, not globally.

### ⚡ Performance

- [ ] Replace `sentence-transformers` with **Ollama embedding models** (`nomic-embed-text`) for tighter local integration.
- [ ] Vector DB backend (e.g., **Chroma**, **Weaviate**, or **FAISS**) instead of raw caching.
- [ ] Async Ollama client for reduced latency.

### 🔌 MCP & Tools

- [ ] Real **web search integration** (DuckDuckGo, Brave, or local index).
- [ ] Expand calculator (unit conversion, datetime math, etc.).
- [ ] File-system document store (search across indexed PDFs, docs, notes).
- [ ] Custom tool plugins (weather, APIs, local DB queries).

### 🎨 UI

- [ ] Dark mode & theming for Streamlit.
- [ ] Upload & manage documents in a library view.
- [ ] Conversation tagging, filtering, and export (Markdown / JSON).
- [ ] Chart & table rendering in responses.

### 🗄️ Storage

- [ ] Support **Postgres** for multi-user persistence.
- [ ] Export conversation history as Markdown/HTML.
- [ ] Fine-grained logging of RAG/MCP context used per message.

---

## 🤝 Contributing

Pull requests welcome! Open an issue to discuss new features, improvements, or bug fixes.

---

## 📜 License

MIT License. Free for personal and commercial use.
