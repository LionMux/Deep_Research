"""
Outline-guided synthesis — the core of SciRAG.
Generates research outline from query using LLM API.
Now with TreeNode support for recursive gap_critic.
"""

import json
import logging
from typing import List, Dict, Optional

from .llm_client import KimiClient
from .config import SciRAGConfig
from .tree_node import TreeNode

logger = logging.getLogger(__name__)


class OutlineGenerator:
    """Generates research outline from user query via LLM."""

    SYSTEM_PROMPT = """You are an expert academic research assistant.
Your task is to decompose a research query into a structured outline for a literature review.

CRITICAL INSTRUCTIONS:
1. EACH section must DIRECTLY address the research query — do NOT drift into related but tangential topics.
2. If the query asks "How is X used for Y", focus on Y USING X, not on X itself.
3. Do NOT include sections about general theory or architecture unless the query specifically asks for it.

For each section, provide:
- title: concise section title
- description: what to cover (2-3 sentences)
- keywords: key terms to search for in papers (3-5 terms)
- percentage: approximate % of total content this section should cover

Respond ONLY in valid JSON — no introductory text, no explanations, no markdown:
{"sections": [{"title": "...", "description": "...", "keywords": ["..."], "percentage": 25}]}"""

    def __init__(self, client: KimiClient, config: SciRAGConfig):
        self.client = client
        self.config = config

    def generate(self, query: str, max_sections: int = 8) -> TreeNode:
        """Generate outline from query. Returns TreeNode root."""
        logger.info(f"Generating outline for query: {query}")

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Create a research outline (max {max_sections} sections) for:\n{query}\n\nRespond in JSON."}
        ]

        result = self.client.chat_json(
            messages,
            model=self.config.kimi_model_outline,
            temperature=self.config.kimi_temperature,
            max_tokens=2048,
        )

        sections = result.get("sections", [])
        if not sections and "raw" in result:
            sections = self._parse_raw_outline(result["raw"])

        # Build TreeNode tree
        root = TreeNode(title="Research Synthesis", description=query, depth=0)
        for i, sec in enumerate(sections[:max_sections]):
            child = root.add_child(
                title=sec.get("title", f"Section {i + 1}"),
                description=sec.get("description", ""),
                keywords=sec.get("keywords", [])
            )
            # Store percentage if provided
            child.value = sec.get("percentage", 0) / 100.0
            logger.debug(f"  Section: {child.title} — {child.keywords} ({child.value:.0%})")

        logger.info(f"Outline generated: {len(root.children)} sections")
        return root

    def critique_and_expand(self, root: TreeNode, query: str) -> TreeNode:
        """Critique outline for gaps and expand if needed (SciRAG 'plan-critique-solve')."""
        logger.info("Critiquing outline for gaps...")

        outline_text = "\n".join(
            f"{i+1}. {n.title}: {n.description} ({n.value:.0%})"
            for i, n in enumerate(root.children)
        )

        messages = [
            {"role": "system", "content": "You are a critical reviewer. Identify gaps in this research outline. What important aspects are missing? Respond in JSON: {\"gaps\": [{\"title\":\"...\",\"description\":\"...\",\"keywords\":[\"...\"]}]}"},
            {"role": "user", "content": f"Query: {query}\n\nOutline:\n{outline_text}\n\nWhat is missing?"}
        ]

        result = self.client.chat_json(messages, model=self.config.kimi_model_outline)
        gaps = result.get("gaps", [])

        for gap in gaps:
            child = root.add_child(
                title=gap.get("title", "Missing Section"),
                description=gap.get("description", ""),
                keywords=gap.get("keywords", [])
            )
            child.depth = 1
            logger.debug(f"  Added gap section: {child.title}")

        logger.info(f"Outline expanded: {len(gaps)} gaps filled, total {len(root.children)} sections")
        return root

    def _parse_raw_outline(self, raw: str) -> List[Dict]:
        """Fallback parser for non-JSON responses."""
        import re
        sections = []
        lines = raw.split("\n")
        current = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            m = re.match(r'^(?:\d+[.\-]\s*)?(.+?)[:\-\–]\s*(.*)', line)
            if m:
                if current:
                    sections.append(current)
                current = {"title": m.group(1).strip(), "description": m.group(2).strip(), "keywords": []}
        if current:
            sections.append(current)
        return sections