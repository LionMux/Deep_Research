"""
TreeNode — recursive tree structure for SciRAG gap_critic.

Inspired by Princeton NLP "tree-of-thought-llm" (BFS exploration)
and Yale NLP SciRAG (TreeNode with parent/children).

Each node = one section in the research outline.
Parent/children relationships form a tree for:
  - depth-limited critique
  - bottom-up aggregation
  - parallel gap analysis (ThreadPoolExecutor)
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


class TreeNode:
    """
    One node in the research outline tree.
    Mirrors SciRAG TreeNode: title, content, parent, children, value.
    """

    _id_counter = 0

    def __init__(self, title: str, description: str = "", depth: int = 0,
                 parent: Optional["TreeNode"] = None, keywords: Optional[List[str]] = None):
        TreeNode._id_counter += 1
        self.id = f"node_{TreeNode._id_counter}"
        self.title = title
        self.description = description
        self.depth = depth
        self.parent = parent
        self.keywords = keywords or []
        self.children: List[TreeNode] = []

        # SciRAG-specific fields
        self.facts: List[Dict] = []          # extracted facts
        self.draft: str = ""                # generated text
        self.citations: List[str] = []       # [Karpukhin2020, Sec. 3]
        self.confidence: float = 0.0         # verification score
        self.value: float = 0.0            # LLM-evaluated quality score [0,1]
        self.gaps: List[str] = []           # critique feedback
        self.is_leaf: bool = True          # no children yet
        self.is_complete: bool = False      # finished synthesis

    def add_child(self, title: str, description: str = "", keywords: Optional[List[str]] = None) -> "TreeNode":
        """Add a child node."""
        child = TreeNode(title=title, description=description,
                         depth=self.depth + 1, parent=self, keywords=keywords)
        self.children.append(child)
        self.is_leaf = False
        return child

    def get_path(self) -> List["TreeNode"]:
        """Get path from root to this node."""
        path = []
        node = self
        while node:
            path.insert(0, node)
            node = node.parent
        return path

    def get_siblings(self) -> List["TreeNode"]:
        """Get sibling nodes (same parent)."""
        if not self.parent:
            return []
        return [c for c in self.parent.children if c != self]

    def get_leaves(self) -> List["TreeNode"]:
        """Get all leaf descendants."""
        if self.is_leaf:
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "depth": self.depth,
            "keywords": self.keywords,
            "facts_count": len(self.facts),
            "draft_preview": self.draft[:200],
            "confidence": round(self.confidence, 2),
            "value": round(self.value, 2),
            "gaps": self.gaps,
            "is_leaf": self.is_leaf,
            "is_complete": self.is_complete,
            "children_count": len(self.children),
            "parent": self.parent.id if self.parent else None,
        }

    def __repr__(self):
        return f"TreeNode({self.id}: {self.title}, depth={self.depth}, value={self.value:.2f})"


class GapCriticTree:
    """
    Recursive gap critic with tree traversal.
    
    Algorithm (BFS, inspired by tree-of-thought-llm):
      1. Build tree from outline
      2. For each leaf: generate draft → evaluate value
      3. Criticize: identify gaps → if gaps found, add child nodes
      4. Repeat until max_depth or no gaps
      5. Bottom-up: merge children → parent
    """

    def __init__(self, llm_generate: Callable, llm_evaluate: Callable,
                 max_depth: int = 2, max_gaps_per_node: int = 2):
        """
        llm_generate: fn(prompt) → text — generates draft for a node
        llm_evaluate: fn(prompt, draft) → float [0,1] — evaluates quality
        """
        self.llm_generate = llm_generate
        self.llm_evaluate = llm_evaluate
        self.max_depth = max_depth
        self.max_gaps_per_node = max_gaps_per_node

    def build_tree(self, outline_sections: List[Dict]) -> TreeNode:
        """Build tree from outline sections."""
        root = TreeNode(title="Research Synthesis", description="Root node")

        for sec in outline_sections:
            child = root.add_child(
                title=sec.get("title", "Untitled"),
                description=sec.get("description", ""),
                keywords=sec.get("keywords", [])
            )
            logger.debug(f"Added child: {child.title} (depth={child.depth})")

        logger.info(f"GapCriticTree built: {len(root.children)} top-level sections")
        return root

    def critique_and_expand(self, root: TreeNode, context: str = "",
                            parallel: bool = True) -> TreeNode:
        """
        BFS critique: critique each leaf, expand if gaps found.
        
        Args:
            root: TreeNode root
            context: additional context for generation
            parallel: use ThreadPoolExecutor for parallel critique
        
        Returns:
            Expanded tree with all gaps filled or depth limit reached
        """
        logger.info("Starting recursive gap critique...")

        for iteration in range(self.max_depth + 1):
            # Find all incomplete leaves within depth limit
            leaves = [n for n in root.get_leaves()
                      if n.depth <= self.max_depth and not n.is_complete]

            if not leaves:
                logger.info(f"All nodes complete at iteration {iteration}")
                break

            logger.info(f"Iteration {iteration}: {len(leaves)} nodes to critique")

            if parallel and len(leaves) > 1:
                self._critique_parallel(leaves, context)
            else:
                for node in leaves:
                    self._critique_single(node, context)

            # Check if any gaps triggered expansion
            expanded = sum(1 for n in leaves if n.gaps)
            if expanded == 0:
                logger.info("No gaps found — stopping early")
                break

        # Mark all leaves as complete
        for leaf in root.get_leaves():
            leaf.is_complete = True

        logger.info("Gap critique complete")
        return root

    def _critique_single(self, node: TreeNode, context: str):
        """Critique one node and expand if gaps found."""
        # Generate draft if empty
        if not node.draft:
            node.draft = self.llm_generate(node, context)

        # Evaluate quality
        node.value = self.llm_evaluate(node, context)
        node.confidence = node.value

        # Critique
        prompt = self._build_critique_prompt(node)
        critique_text = self.llm_generate(None, prompt)  # None = raw prompt mode

        gaps = self._parse_gaps(critique_text)
        node.gaps = gaps[:self.max_gaps_per_node]

        # Expand: add child nodes for each gap
        for gap in node.gaps:
            child = node.add_child(
                title=f"{node.title} — {gap[:40]}",
                description=gap,
                keywords=node.keywords + ["gap", "missing"]
            )
            logger.debug(f"  Expanded: {child.title}")

    def _critique_parallel(self, nodes: List[TreeNode], context: str):
        """Critique multiple nodes in parallel."""
        with ThreadPoolExecutor(max_workers=min(len(nodes), 4)) as executor:
            futures = {executor.submit(self._critique_single, n, context): n for n in nodes}
            for future in as_completed(futures):
                node = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.warning(f"Critique failed for {node.id}: {e}")

    def _build_critique_prompt(self, node: TreeNode) -> str:
        """Build critique prompt for a node."""
        return f"""Critique this academic section for gaps and missing information.

Section Title: {node.title}
Section Description: {node.description}
Draft:
{node.draft[:1000]}

Identify up to {self.max_gaps_per_node} gaps. For each gap, provide:
1. What information is missing
2. What search query would find it

Format: "GAPS: gap1 | gap2 | ..." or "NO GAPS" if complete."""

    def _parse_gaps(self, text: str) -> List[str]:
        """Parse gaps from LLM response."""
        if "NO GAPS" in text.upper():
            return []

        # Try "GAPS: gap1 | gap2" format
        if "GAPS:" in text.upper():
            parts = text.split("GAPS:", 1)[1].split("|")
            return [g.strip() for g in parts if g.strip()]

        # Fallback: split by newlines, filter short lines
        lines = [l.strip("- ") for l in text.split("\n") if l.strip()]
        return [l for l in lines if len(l) > 20 and len(l) < 200][:self.max_gaps_per_node]

    def bottom_up_aggregate(self, root: TreeNode) -> str:
        """
        Bottom-up aggregation: merge children into parent.
        
        SciRAG algorithm:
          1. Process leaves (generate drafts)
          2. Merge siblings into parent
          3. Repeat until root
        """
        logger.info("Starting bottom-up aggregation...")

        def _aggregate_node(node: TreeNode) -> str:
            """Aggregate one node: merge children or return own draft."""
            if node.is_leaf:
                return node.draft or ""

            # Aggregate all children first (post-order)
            child_texts = []
            for child in node.children:
                child_text = _aggregate_node(child)
                child_texts.append(f"### {child.title}\n\n{child_text}")

            # Merge into parent
            merged = f"## {node.title}\n\n{node.description}\n\n"
            merged += "\n\n".join(child_texts)
            node.draft = merged
            return merged

        final = _aggregate_node(root)
        logger.info(f"Aggregation complete: {len(final)} chars")
        return final

    def get_all_nodes(self, root: TreeNode) -> List[TreeNode]:
        """Get all nodes in tree (BFS order)."""
        result = []
        queue = [root]
        while queue:
            node = queue.pop(0)
            result.append(node)
            queue.extend(node.children)
        return result

    def tree_stats(self, root: TreeNode) -> dict:
        """Get tree statistics."""
        all_nodes = self.get_all_nodes(root)
        leaves = [n for n in all_nodes if n.is_leaf]
        complete = [n for n in all_nodes if n.is_complete]
        with_gaps = [n for n in all_nodes if n.gaps]

        return {
            "total_nodes": len(all_nodes),
            "max_depth": max(n.depth for n in all_nodes),
            "leaves": len(leaves),
            "complete": len(complete),
            "with_gaps": len(with_gaps),
            "avg_confidence": round(sum(n.confidence for n in leaves) / max(len(leaves), 1), 2),
        }
