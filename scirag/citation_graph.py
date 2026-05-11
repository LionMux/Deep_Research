"""Citation graph for multi-hop reasoning. Pure Python, CPU only."""

import logging
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import networkx as nx
except ImportError:
    nx = None
    logging.warning("networkx not installed. Run: pip install networkx")

logger = logging.getLogger(__name__)


class CitationGraph:
    """
    Graph of paper citations for SciRAG exploration strategies:
      - backward: cited papers (foundations)
      - forward: citing papers (applications)
      - sequential: deep dive along citation chain
      - parallel: independent sub-questions
    """

    def __init__(self, storage_path: str = "./scirag_data/citation_graph.pkl"):
        self.storage_path = storage_path
        self._graph = nx.DiGraph() if nx else None
        self._paper_meta: Dict[str, dict] = {}  # doc_id → metadata
        self._build_count = 0

    # ---- Building the graph ----

    def add_paper(self, doc_id: str, title: str, year: int = 0,
                  authors: Optional[List[str]] = None,
                  citations_out: Optional[List[str]] = None,
                  citations_in: Optional[List[str]] = None,
                  abstract: str = ""):
        """Add a paper node with metadata."""
        if self._graph is None:
            return
        self._paper_meta[doc_id] = {
            "title": title, "year": year, "authors": authors or [],
            "abstract": abstract,
        }
        self._graph.add_node(doc_id, title=title, year=year)
        for cited in (citations_out or []):
            self._graph.add_edge(doc_id, cited, relation="cites")
        for citing in (citations_in or []):
            self._graph.add_edge(citing, doc_id, relation="cites")

    def build_from_chunks(self, chunks: List[dict]):
        """Build graph from chunk metadata containing citation info."""
        logger.info(f"Building citation graph from {len(chunks)} chunks...")
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            doc_id = chunk.get("document_id", "")
            if not doc_id:
                continue
            # Parse bibliography if present
            bib = meta.get("bibliography", [])
            refs = meta.get("references", [])
            self.add_paper(
                doc_id=doc_id,
                title=meta.get("title", doc_id),
                year=meta.get("year", 0),
                authors=meta.get("authors", []),
                citations_out=bib + refs,
                abstract=meta.get("abstract", ""),
            )
        self._build_count = len(self._paper_meta)
        logger.info(f"Graph built: {self._build_count} papers, {self._graph.number_of_edges()} edges")

    def extract_citations_from_text(self, text: str, doc_id: str,
                                    citation_pattern: Optional[str] = None):
        """Extract citation markers [Author2020] or [1] from text."""
        if citation_pattern is None:
            # Match [Author2020], [Author et al. 2020], [1], [42]
            citation_pattern = r'\[([A-Z][a-z]+(?:\s+et\s+al\.?)?\s*(?:\d{4})?|\d{1,3})\]'
        matches = re.findall(citation_pattern, text)
        for cite_key in matches:
            cited_id = f"{doc_id}_ref_{cite_key}" if cite_key.isdigit() else cite_key
            if self._graph:
                self._graph.add_edge(doc_id, cited_id, relation="cites")

    # ---- Exploration strategies ----

    def explore_backward(self, doc_id: str, depth: int = 2) -> List[str]:
        """Get papers cited BY this paper (foundations)."""
        if self._graph is None or doc_id not in self._graph:
            return []
        result = set()
        frontier = {doc_id}
        for _ in range(depth):
            new_frontier = set()
            for node in frontier:
                for succ in self._graph.successors(node):
                    result.add(succ)
                    new_frontier.add(succ)
            frontier = new_frontier
        return list(result)

    def explore_forward(self, doc_id: str, depth: int = 2) -> List[str]:
        """Get papers citing this paper (applications/extensions)."""
        if self._graph is None or doc_id not in self._graph:
            return []
        result = set()
        frontier = {doc_id}
        for _ in range(depth):
            new_frontier = set()
            for node in frontier:
                for pred in self._graph.predecessors(node):
                    result.add(pred)
                    new_frontier.add(pred)
            frontier = new_frontier
        return list(result)

    def explore_sequential(self, doc_id: str, max_papers: int = 5) -> List[str]:
        """
        Sequential exploration: follow citation chain depth-first.
        For deep, focused questions.
        """
        backward = self.explore_backward(doc_id, depth=3)
        return backward[:max_papers]

    def explore_parallel(self, doc_ids: List[str], max_per: int = 3) -> List[str]:
        """
        Parallel exploration: independent backward searches from multiple seeds.
        For broad, survey questions.
        """
        result = []
        for doc_id in doc_ids:
            result.extend(self.explore_backward(doc_id, depth=1)[:max_per])
        return list(dict.fromkeys(result))  # preserve order, dedup

    def get_related_papers(self, doc_id: str, n: int = 5) -> List[Tuple[str, float]]:
        """Get papers related through citation graph (Jaccard similarity)."""
        if self._graph is None or doc_id not in self._graph:
            return []
        # Common citations similarity
        doc_cites = set(self._graph.successors(doc_id))
        doc_cited_by = set(self._graph.predecessors(doc_id))
        scores = []
        for other in self._graph.nodes():
            if other == doc_id:
                continue
            other_cites = set(self._graph.successors(other))
            other_cited_by = set(self._graph.predecessors(other))
            # Jaccard on union of citations
            union = (doc_cites | doc_cited_by) | (other_cites | other_cited_by)
            if not union:
                continue
            inter = (doc_cites | doc_cited_by) & (other_cites | other_cited_by)
            score = len(inter) / len(union)
            scores.append((other, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

    # ---- Contribution chain reasoning (SciRAG symbolic) ----

    def build_contribution_chain(self, start_doc: str, end_doc: str) -> Optional[List[str]]:
        """
        Find chain: start_doc → ... → end_doc showing intellectual lineage.
        Example: [BERT]T → [DPR]M → [OpenQA]E  (theory → method → experiment)
        """
        if self._graph is None:
            return None
        try:
            path = nx.shortest_path(self._graph, source=start_doc, target=end_doc)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    # ---- Persistence ----

    def save(self):
        if self._graph is None:
            return
        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "wb") as f:
            pickle.dump({"graph": self._graph, "meta": self._paper_meta}, f)
        logger.info(f"Graph saved: {self.storage_path}")

    def load(self) -> bool:
        path = Path(self.storage_path)
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._graph = data["graph"]
            self._paper_meta = data["meta"]
            logger.info(f"Graph loaded: {len(self._paper_meta)} papers")
            return True
        except Exception as e:
            logger.warning(f"Failed to load graph: {e}")
            return False

    @property
    def stats(self) -> dict:
        if self._graph is None:
            return {"status": "networkx not installed"}
        return {
            "papers": self._graph.number_of_nodes(),
            "citations": self._graph.number_of_edges(),
            "density": round(nx.density(self._graph), 4) if nx else 0,
        }
