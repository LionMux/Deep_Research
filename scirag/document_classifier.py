"""
Document type classifier — SciRAG symbolic reasoning component.
Tags documents as: T (Theory), E (Experiment), M (Method), A (Application).
Pure keyword + lightweight embedding, CPU only.
"""

import logging
from collections import Counter
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """
    Classifies text chunks into scientific contribution types.
    Uses keyword matching (fast) + optional embedding similarity (more accurate).
    """

    TAG_DESCRIPTIONS = {
        "T": "Theory — theorems, proofs, definitions, frameworks, bounds, convergence guarantees",
        "E": "Experiment — evaluations, benchmarks, datasets, metrics, ablation studies",
        "M": "Method — algorithms, architectures, techniques, training procedures, pipelines",
        "A": "Application — real-world deployments, case studies, products, systems",
    }

    def __init__(self, keywords: Optional[Dict[str, List[str]]] = None,
                 embedding_model: Optional[str] = None):
        self.keywords = keywords or {
            "T": ["theorem", "proof", "lemma", "proposition", "definition",
                  "theoretical", "framework", "bounds", "convergence",
                  "guarantee", "upper bound", "lower bound", "asymptotic",
                  "generalization bound", "complexity", "analysis"],
            "E": ["experiment", "experimental", "evaluation", "benchmark",
                  "dataset", "results", "accuracy", "precision", "recall",
                  "f1 score", "bleu", "rouge", "throughput", "latency",
                  "ablation", "compared to", "outperforms", "baseline",
                  "metric", "human evaluation", "performance"],
            "M": ["method", "approach", "algorithm", "architecture", "pipeline",
                  "procedure", "technique", "strategy", "training", "inference",
                  "optimization", "loss function", "encoder", "decoder",
                  "attention mechanism", "retrieval model", "embedding",
                  "fine-tune", "pre-train", "model design"],
            "A": ["application", "deploy", "real-world", "industry", "use case",
                  "system", "product", "service", "case study", "practical",
                  "implementation", "production", "scalability", "efficiency",
                  "cost-effective", "user study", "field study"],
        }
        self._embedding = None
        self._tag_embeddings = None

        # Optional: use sentence-transformers for embedding-based classification
        if embedding_model:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding = SentenceTransformer(embedding_model, device="cpu")
                # Precompute tag description embeddings
                tag_texts = [self.TAG_DESCRIPTIONS[t] for t in ["T", "E", "M", "A"]]
                import numpy as np
                self._tag_embeddings = self._embedding.encode(tag_texts, convert_to_numpy=True)
                logger.info(f"Classifier: using embedding model {embedding_model}")
            except Exception as e:
                logger.warning(f"Embedding model not available: {e}")

    def classify(self, text: str, top_k: int = 2) -> List[Dict]:
        """
        Classify text. Returns sorted list of {tag, score, method}.
        Example: [{"tag": "M", "score": 0.85, "method": "keyword"}, ...]
        """
        text_lower = text.lower()
        scores = Counter()

        # Method 1: Keyword matching
        for tag, words in self.keywords.items():
            score = sum(1 for w in words if w in text_lower)
            scores[tag] += score * 0.5  # weight keyword method

        # Method 2: Embedding similarity (if available)
        if self._embedding is not None and self._tag_embeddings is not None:
            import numpy as np
            text_emb = self._embedding.encode([text], convert_to_numpy=True)[0]
            # Cosine similarity
            sims = np.dot(self._tag_embeddings, text_emb) / (
                np.linalg.norm(self._tag_embeddings, axis=1) * np.linalg.norm(text_emb)
            )
            for i, tag in enumerate(["T", "E", "M", "A"]):
                scores[tag] += max(0, sims[i]) * 0.5  # weight embedding method

        # Normalize to [0, 1]
        max_score = max(scores.values()) if scores else 1
        if max_score == 0:
            max_score = 1

        results = []
        for tag in ["T", "E", "M", "A"]:
            results.append({
                "tag": tag,
                "score": round(scores[tag] / max_score, 3),
                "description": self.TAG_DESCRIPTIONS[tag],
                "method": "hybrid" if self._embedding else "keyword",
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def classify_batch(self, texts: List[str]) -> List[List[Dict]]:
        """Classify multiple texts."""
        return [self.classify(t) for t in texts]

    def get_primary_tag(self, text: str) -> str:
        """Get single primary tag for a text."""
        results = self.classify(text, top_k=1)
        return results[0]["tag"] if results else "M"
