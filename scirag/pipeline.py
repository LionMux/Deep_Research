"""
SciRAG Pipeline — entry point.
Orchestrates: outline → retrieval → extraction → classification → synthesis → verification.
Now with TreeNode + recursive gap_critic + bottom-up aggregation.

Usage:
    from scirag import SciRAGPipeline, SciRAGConfig

    config = SciRAGConfig(kimi_api_key="your-key")
    pipeline = SciRAGPipeline(config)

    result = pipeline.run(
        query="neural retrieval methods for information retrieval",
        chunks=my_chunks,
    )
    print(result["final_text"])
"""

import logging
import time
from typing import Dict, List, Optional

from .config import SciRAGConfig
from .llm_client import KimiClient
from .outline_generator import OutlineGenerator
from .tree_node import TreeNode, GapCriticTree
from .citation_graph import CitationGraph
from .document_classifier import DocumentClassifier
from .iterative_synthesizer import IterativeSynthesizer
from .verifier import FactVerifier

# Lazy import to avoid circular dependency during package init
_symbolic_reasoner = None
_s2_client = None

def _get_symbolic_reasoner(client, config):
    global _symbolic_reasoner
    if _symbolic_reasoner is None:
        from .symbolic_reasoning import SymbolicReasoner
        _symbolic_reasoner = SymbolicReasoner
    return _symbolic_reasoner(client, config)

def _get_s2_client():
    global _s2_client
    if _s2_client is None:
        from .s2_client import SemanticScholarClient
        _s2_client = SemanticScholarClient
    return _s2_client()

logger = logging.getLogger(__name__)


class SciRAGPipeline:
    """
    End-to-end SciRAG pipeline with recursive tree synthesis.
    All ML/retrieval runs locally on CPU; only LLM calls go to Kimi API.
    """

    def __init__(self, client: Optional[KimiClient] = None, config: Optional[SciRAGConfig] = None, citation_graph: Optional[CitationGraph] = None):
        self.config = config or SciRAGConfig()
        self.config.ensure_dirs()

        self.client = client or KimiClient(
            api_key=self.config.kimi_api_key,
            base_url=self.config.kimi_base_url,
        )
        self.outline_gen = OutlineGenerator(self.client, self.config)
        self.citation_graph = citation_graph or CitationGraph(self.config.citation_graph_path)
        self.classifier = DocumentClassifier(
            keywords=self.config.classification_keywords,
        )
        self.synthesizer = IterativeSynthesizer(
            self.client, self.config, self.citation_graph,
            reranker=None,  # Set later via property
        )
        self.verifier = FactVerifier(self.client, self.config)
        # Lazy-loaded components (avoid circular imports)
        self._symbolic_reasoner_instance = None
        self._s2_client_instance = None
        self._reranker_instance = None
        self._attributor_instance = None

        logger.info("SciRAG pipeline initialized (TreeNode v2 + Symbolic + S2 + Reranker + Attribution)")

    @property
    def reranker(self):
        """Lazy init BGE reranker."""
        if self._reranker_instance is None:
            from .reranker import SciRAGReranker
            self._reranker_instance = SciRAGReranker(
                model_name=getattr(self.config, 'reranker_model', 'BAAI/bge-reranker-base'),
                device="cpu",
                batch_size=getattr(self.config, 'reranker_batch_size', 8),
                max_length=512,
                normalize=False,
            )
            # Inject into synthesizer
            self.synthesizer.reranker = self._reranker_instance
        return self._reranker_instance

    @property
    def attributor(self):
        """Lazy init PostHocAttributor."""
        if self._attributor_instance is None:
            from .attribution import PostHocAttributor
            self._attributor_instance = PostHocAttributor(self.client)
        return self._attributor_instance

    def _get_s2(self):
        """Lazy init Semantic Scholar client."""
        if self._s2_client_instance is None:
            self._s2_client_instance = _get_s2_client()
        return self._s2_client_instance

    def run(self, query: str, chunks: List[dict],
            retriever=None,
            max_sections: int = 8,
            deep_search_query: str = "") -> Dict:
        """
        Run full SciRAG pipeline with recursive tree.
        """
        t0 = time.time()
        stages = []

        # Track S2 expansion stats across stages
        s2_stats = {"enabled": False}

        logger.info("=" * 50)
        logger.info("SciRAG Pipeline Starting (All 5 Phases)")
        logger.info(f"Query: {query}")
        logger.info(f"Chunks: {len(chunks)}")
        logger.info("=" * 50)

        # === STAGE -1: Deep Search (optional) ===
        deep_search_stats = {"enabled": False}
        if deep_search_query:
            try:
                from .deep_search import AcademicSearchEngine, PDFDownloader
                logger.info("\n[-1/7] Deep Search: discovering papers...")
                t = time.time()

                engine = AcademicSearchEngine(target=50)
                papers, ds_stats = engine.search_sync(deep_search_query)
                deep_search_stats = {
                    "enabled": True,
                    "query": deep_search_query,
                    "found": ds_stats["unique_count"],
                    "with_pdf": ds_stats["with_pdf"],
                    "time": ds_stats["time"],
                }

                # Download PDFs
                if papers:
                    dl = PDFDownloader(papers_dir="./papers")
                    downloaded = dl.download_sync(papers[:10])
                    deep_search_stats["downloaded"] = len(downloaded)
                    logger.info(f"Deep Search: {len(downloaded)} PDFs downloaded")

                stages.append({"stage": "deep_search", "time": round(time.time() - t, 2), **deep_search_stats})
            except Exception as e:
                logger.warning(f"Deep Search failed: {e}")
                stages.append({"stage": "deep_search", "error": str(e)})

        # === STAGE -1b: Firecrawl deep-research (optional, merged into chunks) ===
        firecrawl_stats = {"enabled": False}
        if getattr(self.config, "enable_firecrawl_deep_research", False):
            try:
                # Avoid hard dependency on third_party package structure
                import importlib.util
                import pathlib
                from .firecrawl_adapter import firecrawl_report_to_chunks

                firecrawl_query = deep_search_query or getattr(self.config, "firecrawl_deep_search_query", "") or query
                t = time.time()
                logger.info("\n[-1b/7] Firecrawl deep-research: generating web evidence...")

                # Load third_party/open-deep-research-w-firecrawl/run_deep_research from coordinator.py
                repo_root = pathlib.Path(__file__).resolve().parent.parent
                coordinator_path = repo_root / "third_party" / "open-deep-research-w-firecrawl" / "coordinator.py"
                if not coordinator_path.exists():
                    raise FileNotFoundError(f"Firecrawl coordinator.py not found: {coordinator_path}")

                spec = importlib.util.spec_from_file_location("firecrawl_coordinator", coordinator_path)
                if spec is None or spec.loader is None:
                    raise RuntimeError("Failed to import Firecrawl coordinator module")

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore

                if not hasattr(module, "run_deep_research"):
                    raise AttributeError("Firecrawl coordinator does not export run_deep_research")

                # NOTE: coordinator.run_deep_research(user_query) returns markdown string
                report_md = module.run_deep_research(firecrawl_query)

                firecrawl_chunks = firecrawl_report_to_chunks(
                    report_markdown=report_md,
                    query=firecrawl_query,
                    max_sections=self.config.firecrawl_max_sections,
                    max_chunks=self.config.firecrawl_max_chunks,
                )

                if firecrawl_chunks:
                    chunks = list(chunks) + firecrawl_chunks
                    firecrawl_stats = {
                        "enabled": True,
                        "query": firecrawl_query,
                        "chunks_added": len(firecrawl_chunks),
                        "time": round(time.time() - t, 2),
                    }
                    logger.info(f"Firecrawl: added {len(firecrawl_chunks)} chunks into evidence pool")
                else:
                    firecrawl_stats = {
                        "enabled": True,
                        "query": firecrawl_query,
                        "chunks_added": 0,
                        "time": round(time.time() - t, 2),
                    }
                    logger.warning("Firecrawl: adapter returned empty chunks")

                stages.append({"stage": "firecrawl_deep_research", **firecrawl_stats})
            except Exception as e:
                logger.warning(f"Firecrawl deep-research failed: {e}")
                stages.append({"stage": "firecrawl_deep_research", "error": str(e)})
        else:
            firecrawl_stats = {"enabled": False}

        # === STAGE 0: Build citation graph ===
        t = time.time()
        self.citation_graph.build_from_chunks(chunks)
        stages.append({"stage": "citation_graph", "time": round(time.time() - t, 2)})

        # === STAGE 0b: Semantic Scholar citation expansion (Phase 3) ===
        t = time.time()
        s2_papers = []
        s2_stats = {"enabled": False}
        try:
            s2 = self._get_s2()
            logger.info("\n[0b/7] Semantic Scholar citation expansion...")
            s2_papers = s2.search_and_expand(
                query=query,
                citation_graph=self.citation_graph,
                search_limit=3,
                backward_depth=1,
                forward_depth=0,
                max_per_level=10,
            )
            s2_stats = {
                "enabled": True,
                "papers_found": len(s2_papers),
                "graph_papers": self.citation_graph._graph.number_of_nodes(),
                "graph_edges": self.citation_graph._graph.number_of_edges(),
                "cache": s2.cache.stats(),
            }
            logger.info(f"S2 expansion: {len(s2_papers)} papers added to graph")
            stages.append({
                "stage": "s2_expansion",
                "time": round(time.time() - t, 2),
                **s2_stats,
            })
        except Exception as e:
            logger.warning(f"S2 expansion failed: {e}")
            stages.append({"stage": "s2_expansion", "time": round(time.time() - t, 2), "error": str(e)})

        # === STAGE 1: Generate outline tree ===
        t = time.time()
        logger.info("\n[1/7] Generating outline tree...")
        root = self.outline_gen.generate(query, max_sections=max_sections)
        root = self.outline_gen.critique_and_expand(root, query)
        stages.append({"stage": "outline_tree", "time": round(time.time() - t, 2),
                       "sections": len(root.children), "tree_depth": 1})
        logger.info(f"Outline tree: {len(root.children)} sections")
        for i, child in enumerate(root.children, 1):
            logger.info(f"  {i}. {child.title} — keywords: {child.keywords} ({child.value:.0%})")

        # === STAGE 1b: Symbolic Reasoning (Phase 2) ===
        t = time.time()
        logger.info("\n[1b/7] Symbolic Reasoning (T/E/A segments + relationships)...")
        
        # Lazy init SymbolicReasoner
        if self._symbolic_reasoner_instance is None:
            self._symbolic_reasoner_instance = _get_symbolic_reasoner(self.client, self.config)
        
        # Convert chunks to papers format for symbolic reasoning
        papers_for_symbolic = []
        seen_doc_ids = set()
        for c in chunks:
            did = c.get("document_id", "")
            if did and did not in seen_doc_ids:
                papers_for_symbolic.append({
                    "id": did,
                    "title": c.get("metadata", {}).get("title", did),
                    "text": c.get("text", ""),
                })
                seen_doc_ids.add(did)
        
        symbolic_result = None
        if papers_for_symbolic:
            try:
                symbolic_result = self._symbolic_reasoner_instance.run(
                    papers=papers_for_symbolic,
                    query=query,
                    outline=root,
                )
                stages.append({
                    "stage": "symbolic_reasoning",
                    "time": round(time.time() - t, 2),
                    "papers": len(papers_for_symbolic),
                    "segments": symbolic_result["stats"]["total_segments"],
                    "relationships": symbolic_result["stats"]["relationships"],
                })
                logger.info(f"Symbolic: {symbolic_result['stats']['total_segments']} segments, {symbolic_result['stats']['relationships']} relationships")
            except Exception as e:
                logger.warning(f"Symbolic reasoning failed: {e}")
                stages.append({"stage": "symbolic_reasoning", "time": round(time.time() - t, 2), "error": str(e)})
        else:
            stages.append({"stage": "symbolic_reasoning", "time": 0, "skipped": True})

        # === STAGE 2: Build retriever ===
        if retriever is None:
            from .embedder import EmbedderFactory
            from .faiss_store import FAISSVectorStore, Chunk
            from .hybrid_retriever import HybridRetriever

            logger.info("\n[2/7] Building local retriever (FAISS-CPU)...")
            t = time.time()

            embedder = EmbedderFactory.create(prefer_bge=False)
            store = FAISSVectorStore()

            chunk_objects = []
            for c in chunks:
                chunk_objects.append(Chunk(
                    id=c.get("id", ""), document_id=c.get("document_id", ""),
                    text=c.get("text", ""), start_pos=c.get("start_pos", 0),
                    end_pos=c.get("end_pos", 0),
                    section_header=c.get("metadata", {}).get("section_header", ""),
                    metadata=c.get("metadata", {}),
                ))

            # Encode and add to store
            texts = [c.text for c in chunk_objects]
            embeddings = embedder.encode(texts)
            store.add_chunks(chunk_objects, embeddings)

            retriever = HybridRetriever(vector_store=store, embedder=embedder)
            stages.append({"stage": "retriever", "time": round(time.time() - t, 2)})
            logger.info("Retriever ready (FAISS-CPU)")

        # === STAGE 3: Recursive synthesis with gap_critic tree ===
        t = time.time()
        logger.info("\n[3/7] Recursive synthesis with gap_critic tree...")
        root = self.synthesizer.synthesize(root, retriever)
        stages.append({"stage": "synthesis", "time": round(time.time() - t, 2),
                       "tree_nodes": len(root.get_leaves())})

        # === STAGE 4: Verify facts ===
        t = time.time()
        logger.info("\n[4/7] Fact verification...")
        all_leaves = root.get_leaves()
        all_facts = [{"text": f.text, "source_doc": f.source_doc}
                     for leaf in all_leaves for f in leaf.facts]

        source_chunks = []
        seen_docs = set()
        for c in chunks:
            if c.get("document_id") not in seen_docs:
                source_chunks.append(c)
                seen_docs.add(c.get("document_id"))

        verify_results = self.verifier.verify_batch(all_facts, source_chunks)

        fact_idx = 0
        for leaf in all_leaves:
            for f in leaf.facts:
                if fact_idx < len(verify_results):
                    v = verify_results[fact_idx]
                    f.verified = v["verified"]
                    f.confidence = v["confidence"]
                    fact_idx += 1

        stages.append({"stage": "verification", "time": round(time.time() - t, 2),
                       "facts": len(all_facts),
                       "verified": sum(1 for r in verify_results if r["verified"])})

        # === STAGE 5: Bottom-up aggregation ===
        t = time.time()
        logger.info("\n[5/7] Bottom-up aggregation...")
        final_text = self.synthesizer.compile_final(root)
        stages.append({"stage": "aggregation", "time": round(time.time() - t, 2)})

        # === STAGE 6: Tree stats ===
        t = time.time()
        logger.info("\n[6/7] Computing tree statistics...")
        
        # Build GapCriticTree just for stats
        gap_critic = GapCriticTree(
            llm_generate=lambda n, ctx: "",
            llm_evaluate=lambda n, ctx: 0.0,
            max_depth=self.config.max_iterations,
        )
        tree_stats = gap_critic.tree_stats(root)
        stages.append({"stage": "stats", "time": round(time.time() - t, 2)})

        # === STAGE 7: Post-hoc attribution ===
        t = time.time()
        logger.info("\n[7/7] Post-hoc citation attribution...")
        attribution_result = None
        try:
            # Collect all source chunks for attribution
            all_source_chunks = []
            for c in chunks:
                all_source_chunks.append({
                    "document_id": c.get("document_id", ""),
                    "text": c.get("text", ""),
                    "title": c.get("title", c.get("document_id", "")),
                })
            attribution_result = self.attributor.attribute_report(
                report_text=final_text,
                chunks=all_source_chunks,
            )
            # Optionally replace final_text with attributed version
            final_text = attribution_result.get("attributed_text", final_text)
            stages.append({
                "stage": "attribution",
                "time": round(time.time() - t, 2),
                "coverage": attribution_result.get("coverage", 0),
                "sentences": attribution_result.get("total_sentences", 0),
                "verified": attribution_result.get("verified_count", 0),
            })
            logger.info(f"Attribution: {attribution_result.get('coverage', 0):.0%} coverage, "
                       f"{attribution_result.get('verified_count', 0)} verified")
        except Exception as e:
            logger.warning(f"Attribution failed: {e}")
            stages.append({"stage": "attribution", "time": round(time.time() - t, 2), "error": str(e)})

        total_time = time.time() - t0
        total_facts = sum(len(leaf.facts) for leaf in all_leaves)
        verified_facts = sum(1 for leaf in all_leaves for f in leaf.facts if f.verified)
        avg_confidence = sum(leaf.confidence for leaf in all_leaves) / max(len(all_leaves), 1)

        result = {
            "query": query,
            "final_text": final_text,
            "tree": {
                "sections": len(root.children),
                "leaves": len(all_leaves),
                "max_depth": tree_stats["max_depth"],
                "avg_confidence": round(tree_stats["avg_confidence"], 2),
                "with_gaps": tree_stats["with_gaps"],
            },
            "symbolic_reasoning": {
                "segments": symbolic_result.get("stats", {}).get("total_segments", 0) if symbolic_result else 0,
                "relationships": symbolic_result.get("stats", {}).get("relationships", 0) if symbolic_result else 0,
                "selections": list(symbolic_result.get("selections", {}).keys()) if symbolic_result else [],
            },
            "s2_expansion": s2_stats if 's2_stats' in dir() else {"enabled": False},
            "deep_search": deep_search_stats,
            "attribution": attribution_result if attribution_result else {"coverage": 0, "enabled": False},
            "stats": {
                "total_time_sec": round(total_time, 1),
                "total_sections": len(root.children),
                "total_leaves": len(all_leaves),
                "total_facts": total_facts,
                "verified_facts": verified_facts,
                "verification_rate": f"{verified_facts}/{total_facts} = {verified_facts/max(total_facts,1)*100:.0f}%",
                "avg_confidence": round(avg_confidence, 2),
                "stages": stages,
            },
        }

        logger.info("\n" + "=" * 50)
        logger.info("SciRAG Complete")
        logger.info(f"Time: {total_time:.1f}s")
        logger.info(f"Tree: {len(root.children)} sections, {len(all_leaves)} leaves, depth {tree_stats['max_depth']}")
        logger.info(f"Facts: {verified_facts}/{total_facts} verified")
        logger.info(f"Confidence: {avg_confidence:.0%}")
        logger.info("=" * 50)

        return result

    def quick_outline(self, query: str) -> TreeNode:
        """Quickly generate outline tree without full synthesis."""
        return self.outline_gen.generate(query)

    def classify_chunks(self, chunks: List[dict]) -> List[Dict]:
        """Classify chunks by T/E/M/A tags."""
        results = []
        for c in chunks:
            tags = self.classifier.classify(c.get("text", ""))
            results.append({
                "document_id": c.get("document_id", ""),
                "primary_tag": tags[0]["tag"] if tags else "M",
                "all_tags": tags,
            })
        return results
