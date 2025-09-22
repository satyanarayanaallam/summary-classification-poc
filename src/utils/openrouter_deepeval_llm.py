from deepeval.models import DeepEvalBaseLLM
from src.utils.ilm_client import OpenRouterILMClient

class OpenRouterDeepevalLLM(DeepEvalBaseLLM):
    def __init__(self, model: str = None):
        self.client = OpenRouterILMClient(model_override=model) if model else OpenRouterILMClient()
        self.model_name = model or self.client.model

    def load_model(self):
        return self.client

    def generate(self, prompt: str, **kwargs) -> str:
        return self.client.generate(prompt)

    async def a_generate(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"OpenRouter/{self.model_name}"
