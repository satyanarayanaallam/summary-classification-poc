import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

class OpenRouterILMClient:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is missing. Please add it to your .env file.")
        self.model = os.getenv("OPENROUTER_MODEL")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def generate(self, prompt: str) -> str:
        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "http://localhost",   # optional, for analytics
                "X-Title": "CrewAI-DeepEval-Integration"
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
