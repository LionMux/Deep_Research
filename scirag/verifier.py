"""
Fact verifier — SciRAG verification layer.
Checks each extracted fact against its source document.
LLM API for semantic verification; local string matching for exact verification.
"""

import logging
from typing import Dict, List

from .llm_client import KimiClient
from .config import SciRAGConfig

logger = logging.getLogger(__name__)


class FactVerifier:
    """
    Verifies facts by checking them against source text.
    Two-pass: (1) local exact match, (2) LLM semantic verification.
    """

    def __init__(self, client: KimiClient, config: SciRAGConfig):
        self.client = client
        self.config = config

    def verify(self, fact_text: str, source_text: str,
               source_doc: str = "") -> Dict:
        """
        Verify one fact against source text.
        Returns: {"verified": bool, "confidence": float, "method": str}
        """
        if not fact_text or not source_text:
            return {"verified": False, "confidence": 0.0, "method": "none"}

        # Pass 1: Local exact/substring match (free, fast)
        local_score = self._local_verify(fact_text, source_text)
        if local_score >= 0.9:
            return {
                "verified": True,
                "confidence": local_score,
                "method": "local_exact",
                "source": source_doc,
            }

        # Pass 2: LLM semantic verification (API call)
        llm_score = self._llm_verify(fact_text, source_text)

        # Combine scores
        final_score = max(local_score, llm_score)
        return {
            "verified": final_score >= self.config.min_confidence,
            "confidence": round(final_score, 3),
            "method": "llm_semantic" if llm_score > local_score else "local_partial",
            "source": source_doc,
        }

    def verify_batch(self, facts: List[Dict], source_chunks: List[dict]) -> List[Dict]:
        """
        Verify multiple facts against source chunks.
        facts: [{"text": "...", "source_doc": "..."}, ...]
        source_chunks: [{"document_id": "...", "text": "..."}, ...]
        """
        # Build source lookup
        sources = {}
        for c in source_chunks:
            doc_id = c.get("document_id", "")
            if doc_id not in sources:
                sources[doc_id] = []
            sources[doc_id].append(c.get("text", ""))

        results = []
        for fact in facts:
            doc_id = fact.get("source_doc", "")
            text = fact.get("text", "")

            # Get source text
            source_texts = sources.get(doc_id, [])
            combined_source = " ".join(source_texts)[:5000]

            result = self.verify(text, combined_source, doc_id)
            results.append(result)

        verified = sum(1 for r in results if r["verified"])
        logger.info(f"Verification: {verified}/{len(results)} facts confirmed")
        return results

    def _local_verify(self, fact: str, source: str) -> float:
        """
        Local verification via substring and keyword overlap.
        Returns confidence [0, 1].
        """
        fact_lower = fact.lower().strip()
        source_lower = source.lower()

        # Exact substring match
        if fact_lower in source_lower:
            return 1.0

        # Keyword overlap (Jaccard)
        fact_words = set(fact_lower.split()) - {"the", "a", "an", "is", "are", "was", "were",
                                                  "in", "on", "at", "to", "for", "of", "with",
                                                  "and", "or", "but", "this", "that", "it"}
        source_words = set(source_lower.split())
        if not fact_words:
            return 0.0

        overlap = len(fact_words & source_words)
        jaccard = overlap / len(fact_words)

        # Number matching: if fact contains numbers, check they're in source
        import re
        fact_numbers = set(re.findall(r'\d+\.?\d*', fact))
        if fact_numbers:
            source_numbers = set(re.findall(r'\d+\.?\d*', source))
            num_match = len(fact_numbers & source_numbers) / len(fact_numbers)
            jaccard = jaccard * 0.5 + num_match * 0.5

        return min(1.0, jaccard)

    def _llm_verify(self, fact: str, source: str) -> float:
        """
        LLM-based semantic verification.
        Ask LLM: is this fact supported by the source text?
        """
        prompt = f"""Does the source text support the following factual claim?

Source text:
{source[:3000]}

Claim to verify:
{fact}

Respond ONLY with a JSON object:
{{"supported": true/false, "confidence": 0.0-1.0, "explanation": "brief reason"}}"""

        messages = [
            {"role": "system", "content": "You are a fact-checking assistant. Determine if a claim is supported by given text."},
            {"role": "user", "content": prompt},
        ]

        result = self.client.chat_json(
            messages,
            model=self.config.kimi_model_verify,
            temperature=0.1,
            max_tokens=512,
        )

        confidence = result.get("confidence", 0.0)
        supported = result.get("supported", False)

        return confidence if supported else confidence * 0.3
