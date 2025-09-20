import requests
import json
import streamlit as st

class OllamaAPI:
    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model
        self.base_url = "http://localhost:11434/api"

    def list_models(self):
        try:
            response = requests.get(f"{self.base_url}/tags")
            response.raise_for_status()
            return [m["name"] for m in response.json()["models"]]
        except Exception as e:
            st.sidebar.error(f"Could not fetch Ollama models: {e}")
            return ["llama3.1:8b"]

    def chat(self, messages, stream: bool = True):
        """Send chat messages to Ollama"""
        url = f"{self.base_url}/chat"

        ollama_messages = []
        for msg in messages:
            if not msg.get("content"):
                continue
            role = "user" if msg["role"] in ["user", "system"] else "assistant"
            ollama_messages.append({"role": role, "content": str(msg["content"])})

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": stream,
        }

        # 👀 Debug: Show payload in sidebar
        st.sidebar.subheader("🔍 Ollama Payload")
        st.sidebar.json(payload)

        try:
            with requests.post(url, json=payload, stream=stream) as r:
                r.raise_for_status()

                if stream:
                    collected = ""
                    for line in r.iter_lines():
                        if not line:
                            continue
                        data = line.decode("utf-8").strip()
                        if not data.startswith("{"):
                            continue
                        msg = json.loads(data)
                        if "message" in msg and "content" in msg["message"]:
                            chunk = msg["message"]["content"]
                            collected += chunk
                            yield chunk
                    return collected
                else:
                    return r.json().get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            yield "Error: Could not connect to Ollama. Run `ollama serve`."
        except Exception as e:
            yield f"Ollama Error: {e}"
