"""SciRAG configuration — lightweight, notebook-friendly."""

import os
from dataclasses import dataclass, field
from typing import Optional

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class SciRAGConfig:
    """All settings for SciRAG. Zero GPU required.

    Naming:
      - PRIMARY_* : primary provider (formerly "Kimi")
      - GEMINI_*  : Gemini fallback
      - *_FALLBACK_* : local fallback (LM Studio / Ollama via OpenAI-compatible endpoint)

    Hugging Face (used for synthesize phase, gated by HF token precheck).
    """

    # === Primary provider ===
    # New env naming: PRIMARY_API_KEY / PRIMARY_BASE_URL
    # Back-compat: KIMI_API_KEY / KIMI_BASE_URL
    primary_api_key: str = field(default_factory=lambda: os.getenv("PRIMARY_API_KEY", os.getenv("KIMI_API_KEY", "")))
    primary_base_url: str = field(
        default_factory=lambda: os.getenv(
            "PRIMARY_BASE_URL",
            # Back-compat: keep old KIMI_BASE_URL if provided, otherwise blank (no hidden defaults)
            os.getenv("KIMI_BASE_URL", ""),
        )
    )

    # === Firecrawl deep-research (third_party/open-deep-research-w-firecrawl) ===
    # Option 1: use Firecrawl as extra evidence chunks (merged into provided `chunks`).
    enable_firecrawl_deep_research: bool = field(
        default_factory=lambda: os.getenv("FIRECRAWL_ENABLE_DEEP_RESEARCH", "false").strip().lower() in ("1", "true", "yes", "on")
    )
    # If not set, pipeline will use `deep_search_query` if provided, else fall back to the main `query`.
    firecrawl_deep_search_query: str = field(default_factory=lambda: os.getenv("FIRECRAWL_DEEP_SEARCH_QUERY", "").strip())
    # Hard limits to control cost/latency
    firecrawl_max_runs: int = field(default_factory=lambda: int(os.getenv("FIRECRAWL_MAX_RUNS", "1")))
    firecrawl_timeout_sec: int = field(default_factory=lambda: int(os.getenv("FIRECRAWL_TIMEOUT_SEC", "180")))
    firecrawl_max_sections: int = field(default_factory=lambda: int(os.getenv("FIRECRAWL_MAX_SECTIONS", "8")))
    firecrawl_max_chunks: int = field(default_factory=lambda: int(os.getenv("FIRECRAWL_MAX_CHUNKS", "12")))

    # Local fallback (only used if configured and/or when primary fails)
    # New env naming: PRIMARY_FALLBACK_URL / PRIMARY_FALLBACK_MODEL
    # Back-compat: KIMI_FALLBACK_URL / KIMI_FALLBACK_MODEL
    primary_fallback_url: str = field(default_factory=lambda: os.getenv("PRIMARY_FALLBACK_URL", os.getenv("KIMI_FALLBACK_URL", "")))
    primary_fallback_model: str = field(default_factory=lambda: os.getenv("PRIMARY_FALLBACK_MODEL", os.getenv("KIMI_FALLBACK_MODEL", "")))

    # Primary models for different phases
    primary_model_outline: str = field(
        default_factory=lambda: os.getenv("PRIMARY_MODEL_OUTLINE", os.getenv("KIMI_MODEL_OUTLINE", ""))
    )
    primary_model_synthesis: str = field(
        default_factory=lambda: os.getenv("PRIMARY_MODEL_SYNTHESIS", os.getenv("KIMI_MODEL_SYNTHESIS", ""))
    )
    primary_model_verify: str = field(
        default_factory=lambda: os.getenv("PRIMARY_MODEL_VERIFY", os.getenv("KIMI_MODEL_VERIFY", ""))
    )

    primary_temperature: float = float(os.getenv("PRIMARY_TEMPERATURE", os.getenv("KIMI_TEMPERATURE", "0.3")))

    # === Gemini fallback ===
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", ""))
    gemini_base_url: str = field(default_factory=lambda: os.getenv("GEMINI_BASE_URL", ""))

    # === Hugging Face (used for synthesize phase) ===
    # We use a HF token for gated access. If the token precheck fails,
    # HF is not attempted and Gemini 3 Flash is used instead.
    hf_api_token: str = field(default_factory=lambda: os.getenv("HF_API_TOKEN", os.getenv("HUGGINGFACE_API_TOKEN", "")))
    hf_base_url: str = field(
        default_factory=lambda: os.getenv(
            "HF_BASE_URL",
            # Default: Hugging Face Inference Providers OpenAI-compatible router
            # (chat only) endpoint base: https://router.huggingface.co/v1
            "https://router.huggingface.co/v1",
        )
    )

    # Endpoint model for synthesize; must be a valid HF Inference model id
    # e.g. "mistralai/Mistral-7B-Instruct-v0.3" or an Inference Endpoint.
    hf_model_synthesis: str = field(default_factory=lambda: os.getenv("HF_MODEL_SYNTHESIS", ""))

    # Precheck gating:
    # - If true, we attempt an HF request with a short prompt during runtime before synthesize.
    # - If precheck fails/throws, HF is skipped for synthesize and Gemini is used.
    hf_precheck_enabled: bool = field(default_factory=lambda: os.getenv("HF_PRECHECK_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on"))

    hf_precheck_timeout_sec: int = int(os.getenv("HF_PRECHECK_TIMEOUT_SEC", "20"))

    # === DeepInfra (OpenAI-compatible) alternative to HF for synthesize phase ===
    # Used when HF inference fails (e.g. 404 gated model / rate limits).
    deepinfra_api_token: str = field(default_factory=lambda: os.getenv("DEEPINFRA_API_TOKEN", os.getenv("DEEPINFRA_TOKEN", "")))
    deepinfra_base_url: str = field(
        default_factory=lambda: os.getenv(
            "DEEPINFRA_BASE_URL",
            # OpenAI-compatible
            "https://api.deepinfra.com/v1/openai",
        )
    )
    # Must match DeepInfra model id, e.g. "deepseek-ai/DeepSeek-V2.5" (example depends on your account)
    deepinfra_model_synthesis: str = field(default_factory=lambda: os.getenv("DEEPINFRA_MODEL_SYNTHESIS", ""))

    # Whether to allow DeepInfra before Gemini/PRIMARY fallbacks
    deepinfra_enabled: bool = field(
        default_factory=lambda: os.getenv("DEEPINFRA_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
    )

    # === Local Embedding (22MB, CPU) ===
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"                     # always CPU

    # === Vector Store (FAISS-CPU) ===
    faiss_index_path: str = "./scirag_data/faiss.index"
    chunk_metadata_path: str = "./scirag_data/chunks.json"

    # === Chunking ===
    chunk_size: int = 512
    chunk_overlap: int = 64

    # === Citation Graph ===
    citation_graph_path: str = "./scirag_data/citation_graph.pkl"

    # === Document Classification (symbolic T/E/M/A) ===
    classification_keywords: dict = field(default_factory=lambda: {
        "T": ["theorem", "proof", "lemma", "proposition", "definition",
              "theory", "framework", "model", "equation", "formula",
              "bounds", "convergence", "guarantee", "upper bound", "lower bound"],
        "E": ["experiment", "experimental", "evaluation", "benchmark",
              "dataset", "result", "accuracy", "precision", "recall", "f1",
              "bleu", "rouge", "throughput", "latency", "ablation",
              "compared to", "outperforms", "baseline", "metric"],
        "M": ["method", "approach", "algorithm", "architecture", "pipeline",
              "procedure", "technique", "strategy", "training", "inference",
              "optimization", "loss function", "encoder", "decoder",
              "attention", "retrieval", "embedding", "fine-tune"],
        "A": ["application", "deploy", "real-world", "industry", "use case",
              "system", "product", "service", "case study", "practical",
              "implementation", "production", "scalability", "cost"],
    })

    # === Reranker (BGE cross-encoder, Phase 4) ===
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"  # 278M params, CPU-friendly
    reranker_batch_size: int = 8
    reranker_max_length: int = 512
    reranker_normalize: bool = False

    # === Synthesis ===
    top_k_per_section: int = 8           # chunks per outline section
    max_iterations: int = 3              # refinement loops
    min_confidence: float = 0.7          # verification threshold

    # === Data directories ===
    data_dir: str = "./scirag_data"

    # ------------------------------------------------------------------
    # Back-compat aliases for older code (pre "primary_*" refactor)
    # ------------------------------------------------------------------
    @property
    def kimi_api_key(self) -> str:
        return self.primary_api_key

    @property
    def kimi_base_url(self) -> str:
        return self.primary_base_url

    @property
    def kimi_model_outline(self) -> str:
        return self.primary_model_outline

    @property
    def kimi_model_synthesis(self) -> str:
        return self.primary_model_synthesis

    @property
    def kimi_model_verify(self) -> str:
        return self.primary_model_verify

    @property
    def kimi_temperature(self) -> float:
        return self.primary_temperature

    @property
    def kimi_fallback_url(self) -> str:
        return self.primary_fallback_url

    @property
    def kimi_fallback_model(self) -> str:
        return self.primary_fallback_model

    def ensure_dirs(self):
        os.makedirs(self.data_dir, exist_ok=True)
