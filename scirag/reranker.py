"""
BGE Reranker v2 Integration — Phase 4.

Cross-encoder reranking for SciRAG retrieval pipeline.
Based on FlagOpen/FlagEmbedding (cloned to third_party/flagembedding).

Key features:
  - Reranks top-k documents from FAISS bi-encoder retrieval
  - Uses transformers.AutoModelForSequenceClassification (inspired by FlagEmbedding)
  - CPU-only inference with conservative batch sizes
  - Lazy model loading (loaded on first use)
  - Model caching via HuggingFace cache

Models:
  - BAAI/bge-reranker-base (278M params) — fastest, default
  - BAAI/bge-reranker-large (560M params) — better quality
  - BAAI/bge-reranker-v2-m3 (568M params) — multilingual, best quality

Usage:
  reranker = SciRAGReranker(model_name="BAAI/bge-reranker-base", device="cpu")
  scores = reranker.rerank(query, documents, top_n=5)
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# FlagEmbedding path for reference (cloned repo)
_FLAG_EMBEDDING_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party", "flagembedding"
)

# Lazy imports — only load when reranker is actually used
_torch = None
_np = None
_AutoModelForSequenceClassification = None
_AutoTokenizer = None


def _load_transformers():
    """Lazy load transformers to avoid heavy import at module level."""
    global _torch, _np, _AutoModelForSequenceClassification, _AutoTokenizer
    if _AutoModelForSequenceClassification is None:
        import numpy as np
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        _np = np
        _torch = torch
        _AutoModelForSequenceClassification = AutoModelForSequenceClassification
        _AutoTokenizer = AutoTokenizer
    return _np, _torch, _AutoModelForSequenceClassification, _AutoTokenizer


class SciRAGReranker:
    """
    Cross-encoder reranker for SciRAG.

    Inspired by FlagEmbedding/FlagEmbedding/inference/reranker/encoder_only/base.py
    Uses AutoModelForSequenceClassification with query-passage pairs.

    Args:
        model_name: HuggingFace model name or local path
        device: "cpu", "cuda", or None (auto)
        batch_size: Inference batch size (default 8 for CPU)
        max_length: Max token length per pair (default 512)
        normalize: Apply sigmoid normalization to scores
        cache_dir: HuggingFace cache directory
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: Optional[str] = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
        normalize: bool = False,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or "cpu"
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize
        self.cache_dir = cache_dir

        self._model = None
        self._tokenizer = None
        self._loaded = False

        logger.info(f"SciRAGReranker initialized: {model_name} (device={device}, batch_size={batch_size})")

    def _load(self):
        """Lazy load model and tokenizer."""
        if self._loaded:
            return

        np, torch, AutoModel, AutoTok = _load_transformers()

        logger.info(f"Loading reranker model: {self.model_name}...")

        self._tokenizer = AutoTok.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )

        self._model.to(self.device)
        self._model.eval()

        # Disable fp16 for CPU
        if self.device == "cpu":
            logger.debug("CPU mode: fp16 disabled")

        self._loaded = True
        logger.info(f"Reranker model loaded successfully")

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid normalization for scores."""
        return 1.0 / (1.0 + float("-inf" if x < -100 else 2.718281828459045 ** (-x)))

    def compute_scores(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """
        Compute relevance scores for query-document pairs.

        Args:
            query: Search query
            documents: List of document texts

        Returns:
            List of float scores (higher = more relevant)
        """
        self._load()

        if not documents:
            return []

        np, torch, _, _ = _load_transformers()

        # Build sentence pairs
        sentence_pairs = [[query, doc] for doc in documents]

        # Pre-tokenize without padding to get lengths
        # (Inspired by FlagEmbedding BaseReranker.compute_score_single_gpu)
        all_inputs = []
        for pair in sentence_pairs:
            # Tokenize query and passage separately
            q_tokens = self._tokenizer(
                pair[0],
                add_special_tokens=False,
                max_length=self.max_length * 3 // 4,
                truncation=True,
                return_attention_mask=False,
            )["input_ids"]
            d_tokens = self._tokenizer(
                pair[1],
                add_special_tokens=False,
                max_length=self.max_length,
                truncation=True,
                return_attention_mask=False,
            )["input_ids"]

            # Combine with [CLS] query [SEP] passage [SEP]
            item = self._tokenizer.prepare_for_model(
                q_tokens,
                d_tokens,
                truncation="only_second",
                max_length=self.max_length,
                padding=False,
            )
            all_inputs.append(item)

        # Sort by length to minimize padding (FlagEmbedding optimization)
        length_sorted_idx = np.argsort([-len(x["input_ids"]) for x in all_inputs])
        all_inputs_sorted = [all_inputs[i] for i in length_sorted_idx]

        # Compute scores in batches with OOM recovery
        batch_size = self.batch_size
        all_scores = []

        while True:
            try:
                # Test batch to check OOM
                test_batch = all_inputs_sorted[:min(len(all_inputs_sorted), batch_size)]
                test_inputs = self._tokenizer.pad(
                    test_batch,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    _ = self._model(**test_inputs, return_dict=True).logits.view(-1).float()
                break  # Test passed
            except RuntimeError:
                batch_size = max(1, batch_size * 3 // 4)
                logger.warning(f"OOM in reranker, reducing batch_size to {batch_size}")
            except Exception:
                batch_size = max(1, batch_size * 3 // 4)

        # Main inference loop
        for start_idx in range(0, len(all_inputs_sorted), batch_size):
            batch = all_inputs_sorted[start_idx:start_idx + batch_size]
            inputs = self._tokenizer.pad(
                batch,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                scores = self._model(**inputs, return_dict=True).logits.view(-1).float()
                all_scores.extend(scores.cpu().numpy().tolist())

        # Restore original order
        all_scores = [all_scores[idx] for idx in np.argsort(length_sorted_idx)]

        # Normalize if requested
        if self.normalize:
            all_scores = [self._sigmoid(s) for s in all_scores]

        return all_scores

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_n: int = 5,
        text_key: str = "text",
    ) -> List[Tuple[Dict, float]]:
        """
        Rerank documents and return top-n with scores.

        Args:
            query: Search query
            documents: List of document dicts (must contain text_key)
            top_n: Number of top documents to return
            text_key: Key for document text in dict

        Returns:
            List of (document_dict, score) tuples, sorted by score descending
        """
        texts = [doc.get(text_key, "") for doc in documents]
        scores = self.compute_scores(query, texts)

        # Pair and sort
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked[:top_n]

    def rerank_chunks(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Rerank retrieved chunks and return top-k.

        Integration point for pipeline.py — takes chunks from FAISS retriever,
        reranks them with cross-encoder, returns best top_k.

        Args:
            query: Search query
            chunks: List of chunk dicts with "text" field
            top_k: Number of chunks to return after reranking

        Returns:
            Reranked chunk list (each with added "rerank_score" field)
        """
        if not chunks:
            return []

        ranked = self.rerank(query, chunks, top_n=len(chunks), text_key="text")

        # Add rerank_score to each chunk and return top_k
        result = []
        for chunk, score in ranked[:top_k]:
            chunk_copy = dict(chunk)
            chunk_copy["rerank_score"] = round(float(score), 4)
            result.append(chunk_copy)

        logger.info(f"Reranked {len(chunks)} chunks -> top-{top_k} (best score: {result[0].get('rerank_score', 0):.3f})")
        return result

    def status(self) -> dict:
        """Return reranker status."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "normalize": self.normalize,
            "loaded": self._loaded,
        }
