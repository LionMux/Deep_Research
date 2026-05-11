"""
Iterative synthesizer — the core engine of SciRAG.
Plan → Search → Extract → Fill → Critique → Refine loop.
Now with TreeNode + recursive gap_critic + bottom-up aggregation.

Inspired by Princeton NLP tree-of-thought-llm (BFS exploration)
and Yale NLP SciRAG (bottom-up aggregation).
"""

import json
import logging
from typing import Dict, List, Optional

from .llm_client import KimiClient
from .config import SciRAGConfig
from .tree_node import TreeNode, GapCriticTree
from .citation_graph import CitationGraph
from .document_classifier import DocumentClassifier

logger = logging.getLogger(__name__)


class Fact:
    """One extracted fact with source attribution."""

    def __init__(self, text: str, source_doc: str, source_section: str = "",
                 page: int = 0, confidence: float = 0.0, verified: bool = False):
        self.text = text
        self.source_doc = source_doc
        self.source_section = source_section
        self.page = page
        self.confidence = confidence
        self.verified = verified

    def to_dict(self) -> dict:
        return {
            "text": self.text[:200],
            "source": self.source_doc,
            "section": self.source_section,
            "verified": self.verified,
        }


class IterativeSynthesizer:
    """
    SciRAG synthesis engine with recursive gap_critic tree.
    
    Algorithm:
      1. Build TreeNode tree from outline
      2. For each leaf: search → extract → fill → critique
      3. If gaps found → expand tree (add child nodes)
      4. Repeat until max_depth or no gaps
      5. Bottom-up: merge children → parent → root
    """

    def __init__(self, client: KimiClient, config: SciRAGConfig,
                 citation_graph: Optional[CitationGraph] = None,
                 symbolic_reasoner: Optional = None,
                 reranker: Optional = None):
        self.client = client
        self.config = config
        self.graph = citation_graph or CitationGraph()
        self.symbolic_reasoner = symbolic_reasoner
        self.reranker = reranker
        self.classifier = DocumentClassifier(
            keywords=config.classification_keywords,
        )

    def synthesize(self, root: TreeNode, retriever) -> TreeNode:
        """
        Full synthesis with recursive gap_critic tree.
        
        Args:
            root: TreeNode root with outline sections as children
            retriever: object with .retrieve(query, top_k) method
        
        Returns:
            Completed tree with all sections synthesized
        """
        logger.info(f"Starting synthesis: {len(root.children)} top-level sections")

        # Step 1: Create GapCriticTree for recursive critique
        # GapCriticTree expects:
        #   - llm_generate(node, ctx):
        #        * when node is TreeNode -> generate draft for that node
        #        * when node is None     -> generate critique text from raw prompt `ctx`
        #
        # Previously, critique calls used `llm_generate(None, prompt)`, but our adapter returned ""
        # causing gaps never to be produced. We fix it by sending `ctx` to the LLM in critique mode.
        gap_critic = GapCriticTree(
            llm_generate=lambda node, ctx: (
                self._generate_draft(node, retriever)
                if isinstance(node, TreeNode)
                else self.client.chat(
                    messages=[{"role": "user", "content": str(ctx)}],
                    model=self.config.kimi_model_outline,
                    temperature=0.1,
                    max_tokens=1024,
                )
            ),
            llm_evaluate=lambda node, ctx: self._evaluate_node(node, retriever) if isinstance(node, TreeNode) else 0.0,
            max_depth=self.config.max_iterations,
            max_gaps_per_node=2,
        )

        # Step 2: Process all leaves (search → extract → fill)
        leaves = root.get_leaves()
        logger.info(f"Processing {len(leaves)} leaf nodes...")
        
        for i, leaf in enumerate(leaves, 1):
            logger.info(f"[{i}/{len(leaves)}] Leaf: {leaf.title}")
            self._process_leaf(leaf, retriever)

        # Step 3: Recursive gap critique + expansion
        logger.info("Running recursive gap critique...")
        gap_critic.critique_and_expand(root, parallel=True)

        # Step 4: Process any newly created leaves from expansion
        new_leaves = [n for n in root.get_leaves() if not n.draft]
        if new_leaves:
            logger.info(f"Processing {len(new_leaves)} expanded leaf nodes...")
            for i, leaf in enumerate(new_leaves, 1):
                self._process_leaf(leaf, retriever)

        # Step 5: Bottom-up aggregation
        logger.info("Bottom-up aggregation...")
        final_text = gap_critic.bottom_up_aggregate(root)
        root.draft = final_text

        # Step 6: Cross-section critique (deduplication)
        self._cross_section_critique(root)

        logger.info(f"Synthesis complete: {len(root.get_leaves())} leaves, tree depth {gap_critic.tree_stats(root)['max_depth']}")
        return root

    def _process_leaf(self, node: TreeNode, retriever):
        """Process one leaf: search → extract → fill → evaluate."""
        # --- Search ---
        query = " ".join(node.keywords) if node.keywords else node.title
        try:
            chunks = retriever.retrieve(query, top_k=self.config.top_k_per_section)
        except Exception as e:
            logger.warning(f"  Retrieval failed for {node.title}: {e}")
            chunks = []

        # Citation graph expansion
        expanded = self._expand_via_citation_graph(chunks, query)
        all_chunks = (chunks + expanded)[:self.config.top_k_per_section * 2]

        # Phase 4: Cross-encoder reranking
        if self.reranker is not None and all_chunks:
            try:
                chunk_dicts = []
                for c in all_chunks:
                    chunk_dicts.append({
                        "text": getattr(c, 'text', str(c)),
                        "document_id": getattr(c, 'document_id', ''),
                        "section": getattr(c, 'section', ''),
                        "page": getattr(c, 'page', 0),
                        "source": c,
                    })
                reranked = self.reranker.rerank_chunks(
                    query=query,
                    chunks=chunk_dicts,
                    top_k=self.config.top_k_per_section,
                )
                # Map back to original chunk objects
                all_chunks = [r["source"] for r in reranked]
                logger.info(f"  Reranked: top-{self.config.top_k_per_section} from {len(chunk_dicts)} candidates")
            except Exception as e:
                logger.warning(f"  Reranking failed: {e}")
                all_chunks = all_chunks[:self.config.top_k_per_section]

        # --- Extract facts ---
        if all_chunks:
            node.facts = self._extract_facts(all_chunks, node)
            node.citations = list(set(f"[{f.source_doc}]" for f in node.facts if f.source_doc))

        # --- Classify ---
        for c in all_chunks:
            tag = self.classifier.get_primary_tag(c.text)
            # Store tag on chunk metadata for later
            if hasattr(c, 'metadata'):
                c.metadata['_tag'] = tag

        # --- Fill draft ---
        node.draft = self._generate_draft(node, retriever)

        # --- Evaluate ---
        node.confidence = self._evaluate_node(node, retriever)
        node.is_complete = True

    def _expand_via_citation_graph(self, chunks, query) -> list:
        """Use citation graph to find additional relevant papers."""
        if self.graph._graph is None or not chunks:
            return []

        doc_ids = list(set(c.document_id for c in chunks[:3]))
        related = self.graph.explore_parallel(doc_ids, max_per=2)
        logger.debug(f"  Graph expansion found: {len(related)} related papers")
        return []

    def _extract_facts(self, chunks, node: TreeNode) -> List[Fact]:
        """Extract structured facts from chunks using LLM API."""
        context_parts = []
        for i, c in enumerate(chunks):
            header = f"[{i+1}] Document: {c.document_id}"
            if hasattr(c, 'section_header') and c.section_header:
                header += f" | Section: {c.section_header}"
            context_parts.append(f"{header}\n{c.text[:800]}")

        context = "\n\n".join(context_parts)

        prompt = f"""Extract key facts relevant to: "{node.title}"
For each fact, provide:
- fact: the factual statement (1-2 sentences)
- source: document identifier
- section: section name if known

Respond in JSON:
{{"facts": [{{"fact": "...", "source": "...", "section": "..."}}]}}

Context:
{context[:4000]}
"""
        messages = [
            {"role": "system", "content": "You extract factual claims from academic text with precise source attribution."},
            {"role": "user", "content": prompt},
        ]

        result = self.client.chat_json(
            messages,
            model=self.config.kimi_model_outline,
            temperature=0.1,
            max_tokens=2048,
        )

        facts = []
        for f in result.get("facts", []):
            facts.append(Fact(
                text=f.get("fact", ""),
                source_doc=f.get("source", ""),
                source_section=f.get("section", ""),
            ))

        logger.debug(f"  Extracted {len(facts)} facts for {node.title}")
        return facts

    def _generate_draft(self, node: TreeNode, retriever) -> str:
        """Generate section text from extracted facts."""
        if not node.facts:
            return f"## {node.title}\n\n[No relevant facts found for this section.]"

        facts_text = "\n".join(
            f"- {f.text} [{f.source_doc}]"
            for f in node.facts[:12]
        )

        prompt = f"""Write an academic section titled "{node.title}".
Use ONLY the provided facts. Do NOT add information not in the facts.
Cite sources using [DocumentID] format.

Facts to use:
{facts_text}

Section description: {node.description}

Write 2-4 paragraphs in academic style."""

        messages = [
            {"role": "system", "content": "You are an academic writer. You synthesize research findings into coherent sections with proper citations."},
            {"role": "user", "content": prompt},
        ]

        text = self.client.chat_synthesize(
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
        )

        return f"## {node.title}\n\n{text}"

    def _evaluate_node(self, node: TreeNode, retriever) -> float:
        """Evaluate node quality using fact count and confidence."""
        if not node.facts:
            return 0.2
        
        # Factors: fact count (0-0.4), draft length (0-0.3), citations (0-0.3)
        fact_score = min(len(node.facts) / 5, 1.0) * 0.4
        length_score = min(len(node.draft) / 500, 1.0) * 0.3
        citation_score = min(len(node.citations) / 3, 1.0) * 0.3
        
        return fact_score + length_score + citation_score

    def _cross_section_critique(self, root: TreeNode):
        """Check for cross-section issues: duplication, contradictions."""
        all_nodes = self._get_all_nodes(root)
        seen_facts = set()
        
        for node in all_nodes:
            for f in node.facts:
                key = f.text[:100]
                if key in seen_facts:
                    f.confidence *= 0.8  # penalize duplicates
                seen_facts.add(key)

        logger.debug(f"Cross-section critique: checked {len(seen_facts)} unique facts")

    def _get_all_nodes(self, root: TreeNode) -> List[TreeNode]:
        """Get all nodes in tree (BFS order)."""
        result = []
        queue = [root]
        while queue:
            node = queue.pop(0)
            result.append(node)
            queue.extend(node.children)
        return result

    def compile_final(self, root: TreeNode) -> str:
        """Compile tree into final document with stats."""
        all_nodes = self._get_all_nodes(root)
        leaves = [n for n in all_nodes if n.is_leaf]
        total_facts = sum(len(n.facts) for n in leaves)
        verified_facts = sum(1 for n in leaves for f in n.facts if f.verified)
        avg_confidence = sum(n.confidence for n in leaves) / max(len(leaves), 1)

        parts = []
        parts.append("# Research Synthesis\n")
        parts.append(f"**Sections:** {len(root.children)}  ")
        parts.append(f"**Sub-sections:** {len(leaves)}  ")
        parts.append(f"**Facts extracted:** {total_facts}  ")
        parts.append(f"**Verified facts:** {verified_facts}  ")
        parts.append(f"**Avg. confidence:** {avg_confidence:.0%}\n")

        # Table of contents
        parts.append("## Contents\n")
        for i, child in enumerate(root.children, 1):
            parts.append(f"{i}. [{child.title}](#section-{i})")
            for j, sub in enumerate(child.children, 1):
                parts.append(f"   {i}.{j}. [{sub.title}](#section-{i}-{j})")
        parts.append("")

        # Sections (bottom-up already merged, use root.draft)
        parts.append(root.draft)

        return "\n\n".join(parts)
