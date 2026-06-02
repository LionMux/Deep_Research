"""
Simple embedder using sentence-transformers.
Replaces vector_store.embedder from old architecture.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class SimpleEmbedder:
    """Lightweight embedder (all-MiniLM-L6-v2, 22MB, CPU)."""

    LOCAL_MODEL_PATH = "/tmp/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dim = 384  # MiniLM-L6-v2 output dimension
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return self._model
        # 1. Try local transformers model first (no network needed)
        try:
            import os

            from transformers import AutoModel, AutoTokenizer
            if os.path.isdir(self.LOCAL_MODEL_PATH):
                logger.info(f"Loading local embedder from {self.LOCAL_MODEL_PATH}")
                self._tokenizer = AutoTokenizer.from_pretrained(self.LOCAL_MODEL_PATH)
                self._model = AutoModel.from_pretrained(self.LOCAL_MODEL_PATH)
                self._model.eval()
                logger.info("Local embedder loaded successfully")
                return self._model
        except Exception as e:
            logger.debug(f"Local transformers model failed: {e}")
        # 2. Try sentence-transformers (may need network)
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedder: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedder loaded")
            return self._model
        except Exception as e:
            logger.warning(f"sentence_transformers failed: {e}, using mock embedder")
            self._model = MockSentenceTransformer(self._dim)
            return self._model

    def _encode_transformers(self, texts: List[str]):
        import numpy as np
        import torch

        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        all_embeddings = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self._tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
            with torch.no_grad():
                model_output = self._model(**encoded)
            embeddings = mean_pooling(model_output, encoded['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.numpy())
        return np.vstack(all_embeddings).tolist()

    def encode(self, texts: List[str]) -> List[List[float]]:
        model = self._load()
        if self._tokenizer is not None:
            return self._encode_transformers(texts)
        embeddings = model.encode(texts)
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return embeddings

    @property
    def dim(self) -> int:
        return self._dim


class MockSentenceTransformer:
    """Mock embedder for testing without sentence-transformers."""

    def __init__(self, dim: int = 384):
        self._dim = dim

    def encode(self, texts, **kwargs):
        import numpy as np
        np.random.seed(42)
        return np.random.randn(len(texts), self._dim).tolist()


class EmbedderFactory:
    """Factory for creating embedders."""

    @staticmethod
    def create(prefer_bge: bool = False):
        if prefer_bge:
            try:
                return SimpleEmbedder("BAAI/bge-small-en-v1.5")
            except Exception:
                pass
        return SimpleEmbedder()
