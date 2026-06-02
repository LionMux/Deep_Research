# SciRAG — Scientific Iterative Research Agent

Outline-guided synthesis with citation-graph reasoning for academic research.

## Features

- **5-Phase Pipeline**: TreeNode outline → Symbolic Reasoning → S2 Citation Expansion → BGE Reranking → Post-Hoc Attribution
- **Hybrid Retrieval**: BM25 (sparse) + dense embeddings fused with Reciprocal Rank Fusion
- **1000+ Documents**: Ingestion, chunking, FAISS-CPU retrieval
- **Citation Graph**: NetworkX-based with Semantic Scholar API expansion
- **Fact Verification**: LLM-based per-sentence verification + deterministic fail-closed claim auditing
- **Structured Reports**: sectioned output with numbered citations, a references list, and confidence flags
- **Evaluation Harness**: reproducible `recall@k` / `precision@k` / `mrr` on a fixed benchmark
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
# Install runtime dependencies
pip install -r requirements.txt

# Configure providers (copy and edit)
cp .env.example .env        # then fill in PRIMARY_API_KEY / GEMINI_API_KEY / etc.

# Run research
python main.py query "What are the latest methods for dense passage retrieval?" --papers-dir ./samples/papers

# Start API server
python main.py api --port 8000

# Start MCP server (for Claude Code / Kimi Code)
python main.py mcp
```

See [`.env.example`](./.env.example) for every supported environment variable.
LLM providers are tried in fallback order (primary → Gemini → HF/DeepInfra → local).

A tiny demo corpus lives in [`samples/papers/`](./samples/papers). Large research
corpora and generated indexes/caches are **not** committed (see `.gitignore`); point
`--papers-dir` at your own directory of `.pdf` / `.md` / `.txt` files.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/index` | POST | Index papers |
| `/search` | POST | Search papers |
| `/query` | POST | Full research pipeline |

## Project Structure

```
scirag/                    # Core package
├── pipeline.py            # End-to-end orchestrator
├── orchestrator.py        # Local-LLM iterative orchestrator
├── outline_generator.py   # Outline tree generation
├── tree_node.py           # TreeNode + GapCriticTree
├── symbolic_reasoning.py  # SPO triples + info-units
├── iterative_synthesizer.py  # Recursive synthesis
├── s2_client.py           # Semantic Scholar API client
├── citation_graph.py      # NetworkX citation graph
├── reranker.py            # BGE cross-encoder reranking
├── attribution.py         # Post-hoc per-sentence citation
├── verifier.py            # Fact verification
├── embedder.py            # Sentence-transformers embeddings
├── faiss_store.py         # FAISS vector store
├── hybrid_retriever.py    # Dense retrieval
├── ingestion.py           # Document ingestion / chunking
├── document_classifier.py # Symbolic T/E/M/A classification
├── theme_materialize.py   # Theme/paper materialization
├── llm_client.py          # PrimaryClient (OpenAI-compatible) + fallbacks
├── local_llm_manager.py   # LM Studio / Ollama management
├── firecrawl_adapter.py   # Firecrawl deep-research adapter
├── config.py              # SciRAGConfig (env-driven)
├── state.py               # Shared runtime state
├── api.py                 # FastAPI app
├── mcp_server.py          # MCP server
├── cli.py                 # `scirag` CLI entry
└── deep_search/           # Academic discovery (OpenAlex/CrossRef/ArXiv/S2)

tests/                     # pytest suite
third_party/               # Vendored upstream repos (not linted)
```

## Development

```bash
# Editable install with dev + api + mcp extras
pip install -e ".[dev,api,mcp]"

# Enable git hooks (ruff lint + hygiene)
pre-commit install

# Common tasks (see Makefile)
make lint     # ruff check .
make format   # ruff check --fix + ruff format
make test     # pytest
make eval     # retrieval evaluation harness (sparse, deterministic)
```

### Evaluation harness

Retrieval quality is measured on a small fixed benchmark
([`benchmarks/retrieval_benchmark.json`](./benchmarks/retrieval_benchmark.json))
so quality is reproducible and regressions are caught:

```bash
python -m scirag.evaluation --benchmark benchmarks/retrieval_benchmark.json \
    --mode sparse --top-k 3
```

Reports `recall@k`, `precision@k`, and `mrr`. `scirag.evaluation` also exposes
`citation_precision` and `faithfulness` over a `VerificationReport` (see
[`scirag/claim_verifier.py`](./scirag/claim_verifier.py)). Sparse mode is
deterministic and needs no API keys or model downloads.

Linting (`ruff`) and the test suite run in CI on every push and pull request
(see [`.github/workflows/ci.yml`](./.github/workflows/ci.yml)). See
[`CONTRIBUTING.md`](./CONTRIBUTING.md) for the contribution workflow.

## License

MIT
