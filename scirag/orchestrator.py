"""Local LLM Orchestrator — coordinates RAG retrieval with local LLM as judge.

Architecture (Variant A):
    User Query -> SciRAG Retrieval -> Local LLM (enough info?)
         ^                                    |
         └──── max 2 additional searches ─────┘

Uses KimiClient in fallback mode (localhost:1234) so no cloud API keys needed.
"""

import logging
import re
from typing import Dict

logger = logging.getLogger(__name__)


class LocalLLMOrchestrator:
    """Orchestrates retrieval + local LLM for deep research."""

    def __init__(self, llm_client, retriever_fn, max_extra_retrievals: int = 2):
        """
        Args:
            llm_client: object with .chat(messages, model, temperature, max_tokens) -> str
            retriever_fn: callable(query: str, top_k: int) -> list of chunks
            max_extra_retrievals: how many times LLM can ask for more info
        """
        self.llm = llm_client
        self.retriever_fn = retriever_fn
        self.max_extra = max_extra_retrievals
        logger.info(f"LocalLLMOrchestrator ready (max extra retrievals: {max_extra_retrievals})")

    def research(self, query: str, top_k: int = 8) -> dict:
        """Run the full research loop."""
        all_chunks = []
        history = []

        # --- Iteration 0: initial retrieval ---
        chunks = self.retriever_fn(query, top_k=top_k)
        all_chunks.extend(chunks)
        history.append({"query": query, "chunks": len(chunks)})
        logger.info(f"Initial retrieval: {len(chunks)} chunks")

        current_query = query

        for iteration in range(self.max_extra + 1):
            # Ask LLM to decide
            decision_prompt = self._build_decision_prompt(current_query, all_chunks, iteration)
            raw_decision = self.llm.chat(
                messages=[{"role": "user", "content": decision_prompt}],
                temperature=0.1,
                max_tokens=200,
            )
            decision = self._parse_decision(raw_decision)
            logger.info(f"Iteration {iteration + 1} decision: {decision['action']}")

            if decision["action"] == "ANSWER":
                # Generate final answer
                final_prompt = self._build_final_prompt(query, all_chunks)
                final_answer = self.llm.chat(
                    messages=[{"role": "user", "content": final_prompt}],
                    temperature=0.3,
                    max_tokens=2048,
                )
                return {
                    "query": query,
                    "answer": final_answer,
                    "iterations": iteration + 1,
                    "total_chunks": len(all_chunks),
                    "history": history,
                }

            elif decision["action"] == "SEARCH" and iteration < self.max_extra:
                follow_up = decision.get("follow_up_query", query)
                logger.info(f"LLM asks for more info: {follow_up}")
                chunks = self.retriever_fn(follow_up, top_k=top_k)
                all_chunks.extend(chunks)
                history.append({"query": follow_up, "chunks": len(chunks)})
                current_query = follow_up
            else:
                logger.warning("Max iterations or unclear decision. Forcing final answer.")
                final_prompt = self._build_final_prompt(query, all_chunks)
                final_answer = self.llm.chat(
                    messages=[{"role": "user", "content": final_prompt}],
                    temperature=0.3,
                    max_tokens=2048,
                )
                return {
                    "query": query,
                    "answer": final_answer,
                    "iterations": iteration + 1,
                    "total_chunks": len(all_chunks),
                    "history": history,
                    "forced": True,
                }

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_decision_prompt(self, query: str, chunks: list, iteration: int) -> str:
        preview = "\n".join(
            f"{i+1}. {self._chunk_text(c)[:350]}"
            for i, c in enumerate(chunks[:6])
        )
        return (
            f"Question: {query}\n\n"
            f"Retrieved passages:\n{preview}\n\n"
            f"Are these passages enough to answer the question well?\n"
            f"Reply with EXACTLY one of these two formats:\n"
            f"ANSWER: yes\n"
            f"or\n"
            f"SEARCH: <specific follow-up query>\n\n"
            f"Be decisive. If the passages cover the topic, say ANSWER: yes."
        )

    def _build_final_prompt(self, query: str, chunks: list) -> str:
        preview = "\n\n".join(
            f"[{i+1}] {self._chunk_text(c)[:500]}"
            for i, c in enumerate(chunks[:10])
        )
        return (
            f"Write a comprehensive markdown answer to the question.\n\n"
            f"Question: {query}\n\n"
            f"Use only the following sources. Cite them as [N].\n\n"
            f"{preview}\n\n"
            f"Answer:"
        )

    def _chunk_text(self, c) -> str:
        if isinstance(c, dict):
            return c.get("text", c.get("text_preview", str(c)))
        return getattr(c, "text", str(c))

    def _parse_decision(self, text: str) -> Dict:
        text = text.strip()
        # Look for SEARCH marker
        m = re.search(r"SEARCH[:\s]*(.+)", text, re.IGNORECASE)
        if m:
            follow_up = m.group(1).strip()
            follow_up = re.sub(r"^[\"']+|[\"']+$", "", follow_up)
            return {"action": "SEARCH", "follow_up_query": follow_up}
        # Look for ANSWER marker
        if re.search(r"ANSWER", text, re.IGNORECASE):
            return {"action": "ANSWER"}
        # Heuristic: if text is very short and contains yes / enough
        if len(text) < 50 and any(w in text.lower() for w in ("yes", "enough", "covered", "sufficient")):
            return {"action": "ANSWER"}
        # Default: assume we need more info on first iteration, answer on last
        return {"action": "SEARCH", "follow_up_query": "more details"}
