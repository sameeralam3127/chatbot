# apis.py
import requests

class ChatAPI:
    def __init__(self, api_name, api_key=None, endpoint=None):
        self.api_name = api_name
        self.api_key = api_key
        self.endpoint = endpoint
        self.client = None

        if api_name == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        elif api_name == "claude":
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)

    def send_message(self, messages):
        if self.api_name == "openai":
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
            )
            return response.choices[0].message.content

        elif self.api_name == "ollama":
            payload = {"model": "llama-3.1-8b", "messages": messages}
            r = requests.post(f"{self.endpoint}/chat", json=payload)
            return r.json().get("response", "")

        elif self.api_name == "claude":
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            response = self.client.completions.create(
                model="claude-2",
                prompt=prompt,
                max_tokens_to_sample=1000
            )
            return response["completion"]

        elif self.api_name == "deepseek":
            return "Deepseek response placeholder"
