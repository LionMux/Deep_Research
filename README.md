# SciRAG — Scientific Iterative Research Agent

Outline-guided synthesis with citation-graph reasoning for academic research.

## Features

- **5-Phase Pipeline**: TreeNode outline → Symbolic Reasoning → S2 Citation Expansion → BGE Reranking → Post-Hoc Attribution
- **1000+ Documents**: Ingestion, chunking, FAISS-CPU retrieval
- **Citation Graph**: NetworkX-based with Semantic Scholar API expansion
- **Fact Verification**: LLM-based per-sentence verification
- **MCP Server**: Claude Code / Kimi Code integration
- **FastAPI**: HTTP API for programmatic access

## Architecture

```
[0] Ingestion → chunking → FAISS index
[0b] S2 Citation Expansion (Semantic Scholar API)
[1] Outline Tree Generation (LLM)
[1b] Symbolic Reasoning (SPO triples + info-units)
[2] FAISS Retrieval (bi-encoder)
[2b] BGE Reranking (cross-encoder)
[3] Recursive Synthesis (gap_critic tree)
[4] Fact Verification
[5] Bottom-up Aggregation
[6] Post-Hoc Attribution per sentence
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Set API key
export MOONSHOT_API_KEY="your_key"

# Run research
python main.py query "What are the latest methods for dense passage retrieval?" --papers-dir ./papers

# Start API server
python main.py api --port 8000

# Start MCP server (for Claude Code / Kimi Code)
python main.py mcp
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/index` | POST | Index papers |
| `/search` | POST | Search papers |
| `/query` | POST | Full research pipeline |

## Project Structure

```
scirag/               # Core package
├── pipeline.py       # Orchestrator (8 stages)
├── tree_node.py      # TreeNode + GapCriticTree
├── symbolic_reasoning.py  # SPO triples + info-units
├── s2_client.py      # Semantic Scholar API
├── reranker.py       # BGE cross-encoder
├── attribution.py    # Post-hoc citation
├── embedder.py       # Sentence transformers
├── faiss_store.py    # FAISS vector store
├── hybrid_retriever.py  # Dense retrieval
├── ingestion.py      # Document ingestion
├── llm_client.py    # Kimi API client
├── config.py         # Configuration
├── verifier.py       # Fact verification
├── citation_graph.py # NetworkX graph
├── api.py            # FastAPI app
├── mcp_server.py    # MCP server
└── cli.py            # CLI entry

tests/                # Tests
├── test_phase2_symbolic.py
├── test_phase3_s2.py
├── test_phase4_reranker.py
└── test_phase5_attribution.py

third_party/          # Cloned repos
├── nlp-contrib-graph/  # Phase 2 (SemEval-2021)
└── semanticscholar/    # Phase 3 (S2 API)

data/                 # Data directory
└── vector_index/     # FAISS index + metadata
```

## License

MIT
