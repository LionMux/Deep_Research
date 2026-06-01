"""
Post-Hoc Attribution — Phase 5.

Sentence-level citation attribution for SciRAG, inspired by OpenScholar.
After generation, every sentence without a citation is matched to its
source passage and assigned a citation number.

Key features:
  - Per-sentence citation assignment
  - LLM-based attribution with passages as evidence
  - Verification of existing citations
  - Integration with SciRAG pipeline (after synthesis)

Inspired by:
  - OpenScholar (Princeton NLP / Nature 2025)
    posthoc_attributions_paragraph_all prompt + insert_attributions logic
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AttributedSentence:
    """A sentence with its source citation."""
    text: str
    citation_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    verified: bool = False


class PostHocAttributor:
    """
    Post-hoc citation attribution for generated text.

    Algorithm (from OpenScholar):
      1. Split text into sentences
      2. Identify sentences without citations
      3. For each: ask LLM "which passage supports this?"
      4. Insert [doc_id] citation
      5. Verify existing citations
    """

    def __init__(self, client):
        self.client = client

    # ========================================================================
    # Prompts (from OpenScholar/instructions.py)
    # ========================================================================

    POSTHOC_PROMPT = """We give you a short paragraph extracted from an answer to a question related to scientific literature, and a set of evidence passages.
Find all of the citation-worthy statements without any citations, and insert citation numbers to the statements that are fully supported by any of the provided citations listed as References.
If none of the passages support the statement, do not insert any citation, and leave the original sentence as is.
If multiple passages provide sufficient support for the statement, you only need to insert one citation, rather than inserting all of them.
Your answer should be marked as [Response_Start] and [Response_End].

Here's an example:
Statement: Language models store rich knowledge in their parameters during pre-training, resulting in their strong performance on many knowledge-intensive tasks. However, such parametric knowledge based generations are often hard to attribute.
References:
[0] Title: Attributed Question Answering Text: Roberts et al. shows that T5 can perform closedbook QA, producing answers based on model parameters.
[1] Title: Reliable LMs with Retrieval Text: Unlike parametric LMs, retrieval-augmented LMs leverage external documents at inference.
[Response_Start]Language models store rich knowledge in their parameters during pre-training, resulting in their strong performance on many knowledge-intensive tasks [0]. However, such parametric knowledge based generations are often hard to attribute [1].[Response_End]

Now, please insert citations to the following text. ##
Statement: {statement}
References:\n{passages}\n"""

    VERIFICATION_PROMPT = """We give you a statement extracted from an answer to a scientific literature question, and a cited evidence passage.
Evaluate if the statement is fully supported by the citation, and answer with 'Yes' (fully supported) or 'No' (not supported).
After the rating, explain why it is / is not fully supported.
Your response should be marked as [Response_Start] and [Response_End].

Statement: {statement}
References:\n{passages}\n"""

    # ========================================================================
    # Core Methods
    # ========================================================================

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (OpenScholar uses nltk.sent_tokenize)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def has_citation(sentence: str) -> bool:
        """Check if sentence already has a citation like [N] or [doc_id]."""
        return bool(re.search(r'\[\d+\]|\[[a-zA-Z0-9_-]+\]', sentence))

    @staticmethod
    def format_passages(passages: List[Dict]) -> str:
        """Format passages for prompt."""
        lines = []
        for i, p in enumerate(passages):
            title = p.get("title", p.get("document_id", f"Doc{i}"))
            text = p.get("text", p.get("content", ""))[:300]
            lines.append(f"[{i}] Title: {title} Text: {text}")
        return "\n".join(lines)

    def _call_llm_attribution(self, sentence: str, passages: List[Dict],
                             temperature: float = 0.1) -> str:
        """Call LLM to attribute a sentence to a passage."""
        passages_text = self.format_passages(passages)
        prompt = self.POSTHOC_PROMPT.format(
            statement=sentence,
            passages=passages_text,
        )
        messages = [
            {"role": "system", "content": "You insert accurate citations into scientific text."},
            {"role": "user", "content": prompt},
        ]
        try:
            result = self.client.chat_json(
                messages,
                model="kimi-latest",  # Will use config default
                temperature=temperature,
                max_tokens=500,
            )
            # Extract from JSON or raw text
            raw = result.get("response", result.get("text", str(result)))
        except Exception:
            # Fallback to simple text completion
            try:
                raw = self.client.chat([
                    {"role": "user", "content": prompt}
                ], temperature=temperature, max_tokens=500)
            except Exception as e2:
                logger.warning(f"LLM attribution failed: {e2}")
                return sentence

        # Extract content between [Response_Start] and [Response_End]
        match = re.search(r'\[Response_Start\](.*?)\[Response_End\]', raw, re.DOTALL)
        if match:
            return match.group(1).strip()
        return raw.strip() or sentence

    def _verify_citation(self, sentence: str, passages: List[Dict]) -> Tuple[bool, str]:
        """Verify if a sentence is supported by its cited passage."""
        passages_text = self.format_passages(passages)
        prompt = self.VERIFICATION_PROMPT.format(
            statement=sentence,
            passages=passages_text,
        )
        messages = [
            {"role": "system", "content": "You verify scientific citations."},
            {"role": "user", "content": prompt},
        ]
        try:
            result = self.client.chat_json(
                messages,
                model="kimi-latest",
                temperature=0.1,
                max_tokens=300,
            )
            raw = result.get("response", result.get("text", str(result)))
        except Exception:
            try:
                raw = self.client.chat(messages, temperature=0.1, max_tokens=300)
            except Exception:
                return False, "Verification failed"

        match = re.search(r'\[Response_Start\](.*?)\[Response_End\]', raw, re.DOTALL)
        if match:
            raw = match.group(1).strip()

        # Parse Yes/No
        is_yes = "yes" in raw.lower()[:50]
        return is_yes, raw

    # ========================================================================
    # Public API
    # ========================================================================

    def attribute_text(
        self,
        text: str,
        passages: List[Dict],
        verify_existing: bool = True,
    ) -> Tuple[str, List[AttributedSentence]]:
        """
        Post-hoc attribution: add citations to uncited sentences.

        Args:
            text: Generated text
            passages: Source passages with "text" and "document_id" fields
            verify_existing: Also verify existing citations

        Returns:
            (attributed_text, list of AttributedSentence)
        """
        sentences = self.split_sentences(text)
        if not sentences:
            return text, []

        attributed_sentences = []
        results = []

        for sent in sentences:
            if len(sent) < 5:
                # Too short, skip
                attributed_sentences.append(AttributedSentence(text=sent))
                results.append(sent)
                continue

            if self.has_citation(sent):
                # Already has citation — optionally verify
                if verify_existing:
                    is_valid, explanation = self._verify_citation(sent, passages)
                    cited_ids = re.findall(r'\[(.*?)\]', sent)
                    attributed_sentences.append(AttributedSentence(
                        text=sent,
                        citation_ids=cited_ids,
                        verified=is_valid,
                    ))
                else:
                    cited_ids = re.findall(r'\[(.*?)\]', sent)
                    attributed_sentences.append(AttributedSentence(
                        text=sent,
                        citation_ids=cited_ids,
                        verified=True,
                    ))
                results.append(sent)
            else:
                # No citation — attribute
                attributed = self._call_llm_attribution(sent, passages)
                cited_ids = re.findall(r'\[(.*?)\]', attributed)
                attributed_sentences.append(AttributedSentence(
                    text=attributed,
                    citation_ids=cited_ids,
                    confidence=0.8 if cited_ids else 0.0,
                ))
                results.append(attributed)

        final_text = " ".join(results)
        logger.info(f"Post-hoc attribution: {len(sentences)} sentences, "
                   f"{sum(1 for a in attributed_sentences if a.citation_ids)} attributed")
        return final_text, attributed_sentences

    def attribute_report(
        self,
        report_text: str,
        chunks: List[Dict],
        citations: List[Dict] = None,
    ) -> Dict:
        """
        Full attribution for a research report.

        Args:
            report_text: Full generated report
            chunks: All source chunks used
            citations: Existing citations if any

        Returns:
            {
                "attributed_text": str,
                "sentences": [AttributedSentence],
                "coverage": float,  # % of sentences with citations
                "verified_count": int,
            }
        """
        attributed_text, sentences = self.attribute_text(
            report_text, chunks, verify_existing=True
        )

        total = len(sentences)
        with_citations = sum(1 for s in sentences if s.citation_ids)
        verified = sum(1 for s in sentences if s.verified)
        coverage = with_citations / total if total > 0 else 0.0

        return {
            "attributed_text": attributed_text,
            "sentences": [
                {
                    "text": s.text[:200],
                    "citations": s.citation_ids,
                    "confidence": s.confidence,
                    "verified": s.verified,
                }
                for s in sentences
            ],
            "coverage": round(coverage, 2),
            "verified_count": verified,
            "total_sentences": total,
        }
