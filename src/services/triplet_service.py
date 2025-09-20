"""Service that ties the LLM client and normalization utilities."""
from src.models.gemini_client import GeminiClient
from src.utils.normalization import normalize_triplet
from typing import List, Tuple


class TripletService:
    def __init__(self):
        self.client = GeminiClient()

    def extract_and_normalize(self, text: str) -> List[Tuple[str, str, str]]:
        raw = self.client.extract_triplets(text)
        return [normalize_triplet(t) for t in raw]
