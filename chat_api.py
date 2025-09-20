# chat_api.py
import requests
import json
import streamlit as st
from typing import List, Dict

class OllamaAPI:
    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434/api"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def list_models(self) -> List[str]:
        """Fetch available local Ollama models and filter out embedding-only models."""
        try:
            resp = requests.get(f"{self.base_url}/tags", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            models = []
            for m in data.get("models", []):
                name = m.get("name")
                if not name:
                    continue
                # filter out obvious embed-only names
                if "embed" in name.lower() or "nomic" in name.lower():
                    continue
                models.append(name)
            return models
        except Exception as e:
            st.sidebar.error(f"Could not fetch Ollama models: {e}")
            return []

    def _clean_messages(self, messages: List[Dict]) -> List[Dict]:
        """Return a list of user/assistant messages only, stripped of MCP/RAG dumps and errors."""
        cleaned = []
        for msg in messages:
            role = msg.get("role", "")
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            # skip explicit error strings or debug blocks
            if "Ollama Error" in content or content.startswith("🔍 Ollama Payload"):
                continue
            # skip system messages and any MCP/RAG debug blocks
            if role == "system":
                continue
            if content.startswith("MCP resources:") or content.startswith("RAG context:"):
                # do not send MCP/RAG dumps. keep only the final line (the user's query) if present
                lines = [l.strip() for l in content.splitlines() if l.strip()]
                if not lines:
                    continue
                # if the last line looks like a user query, keep it; else skip entirely
                content = lines[-1]

            # final guard: ensure roles are only 'user' or 'assistant'
            final_role = "user" if role == "user" else "assistant"
            cleaned.append({"role": final_role, "content": str(content)})
        # dedupe consecutive identical user messages
        deduped = []
        for i, m in enumerate(cleaned):
            if i > 0 and m["role"] == "user" and m["content"] == cleaned[i-1]["content"]:
                continue
            deduped.append(m)
        return deduped

    def chat(self, messages: List[Dict], stream: bool = True):
        """
        Send cleaned chat messages to Ollama's /api/chat and yield chunks if streaming.
        Yields strings when streaming; returns final string on completion (generator + return pattern).
        """
        url = f"{self.base_url}/chat"
        payload = {"model": self.model}

        cleaned = self._clean_messages(messages)
        payload["messages"] = cleaned
        # include stream flag for behavior clarity
        payload["stream"] = stream

        # debug: show minimal payload in sidebar (clean)
        try:
            st.sidebar.subheader("🔍 Ollama Payload (clean)")
            st.sidebar.json(payload)
        except Exception:
            pass

        try:
            with requests.post(url, json=payload, stream=stream, timeout=60) as r:
                r.raise_for_status()
                if stream:
                    collected = ""
                    for raw in r.iter_lines():  # streaming lines
                        if not raw:
                            continue
                        try:
                            decoded = raw.decode("utf-8").strip()
                        except Exception:
                            continue
                        if not decoded.startswith("{"):
                            continue
                        try:
                            chunk_obj = json.loads(decoded)
                        except Exception:
                            continue
                        # Ollama stream format: {"message": {"content": "..."}}
                        message = chunk_obj.get("message") or {}
                        content = message.get("content")
                        if content:
                            collected += content
                            yield content
                    # final return: yield nothing further but allow caller to get full text from appended chat history
                    return
                else:
                    data = r.json()
                    return data.get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            yield "Error: Could not connect to Ollama. Make sure `ollama serve` is running."
        except Exception as e:
            yield f"Ollama Error: {e}"
