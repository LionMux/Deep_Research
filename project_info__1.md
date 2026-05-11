# Hardcoded Values / Configuration Review — SciRAG Codebase

## Summary
This is a Python (SciRAG) repository that builds an evidence base (FAISS + hybrid retrieval) and synthesizes answers using multiple LLM backends (OpenAI-compatible “primary”, Gemini, optional HuggingFace/local providers). During the scan for hardcoded values, the most important findings are *not* random constants, but **embedded defaults** and **fallback URLs/models** that can change runtime behavior even when environment variables are set. There are also a few **absolute filesystem paths** (e.g., for local embedding model weights) that can break in non-identical environments.

## Architecture
- **Core pipeline**: orchestrates ingestion → embedding → retrieval → synthesis (primarily under `scirag/`).
- **LLM abstraction**: `scirag/llm_client.py` routes chat requests to different providers based on config/env.
- **Vector store**: `scirag/faiss_store.py` implements FAISS index management and a NumPy fallback.
- **Retrieval**: `scirag/hybrid_retriever.py` combines FAISS search and other retrieval signals.
- **Local embedding**: `scirag/embedder.py` loads either a HuggingFace model name or a **local path** for sentence embeddings.

## Key Hardcoded / “Implicit Defaults” Found
Below are the hardcoded or effectively-hardcoded values that are most likely to surprise a developer.

### 1) Hardcoded local embedding path
- **File**: `scirag/embedder.py`
- **Hardcoded value(s)**:
  - `LOCAL_MODEL_PATH = "/tmp/all-MiniLM-L6-v2"`
  - `self._dim = 384  # MiniLM-L6-v2 output dimension`
- **Meaning / risk**:
  - This is an **absolute POSIX path**; it will not exist on Windows unless someone creates it.
  - If the directory exists, it changes behavior to “load local embedder” rather than using the configured model name.
- **Why it was likely built this way**:
  - To support offline / repeatable embedding without downloading every time.
- **What to check**:
  - Whether the code has Windows-compatible alternate paths or env overrides (the scan result suggests it may not).

### 2) Hardcoded FAISS index paths (relative defaults)
- **File**: `scirag/config.py`
- **Hardcoded default values**:
  - `faiss_index_path: str = "./scirag_data/faiss.index"`
  - `chunk_metadata_path: str = "./scirag_data/chunks.json"`
- **Meaning / risk**:
  - Even when not explicitly configured, the pipeline will write/read these relative paths from the current working directory.
  - In multi-user or CI contexts, this can cause collisions or stale data.

### 3) Hardcoded local LLM base URL and default model
- **File**: `scirag/local_llm_manager.py`
- **Hardcoded defaults**:
  - `base_url: str = "http://localhost:1234/v1"`
  - `model: str = "qwen2.5-3b-instruct"`
- **Meaning / risk**:
  - If the “local” route is selected (or fallback logic picks it), it will attempt to contact `localhost:1234`.
  - The specific model string may not match whatever the local server actually hosts.

### 4) Hardcoded HTTP endpoint base for HuggingFace router
- **File**: `scirag/config.py`
- **Hardcoded default**:
  - Comment indicates HF router default: `https://router.huggingface.co/v1`
- **Meaning / risk**:
  - If HF config env vars are absent or partially configured, this default can be used and lead to unexpected network calls.

### 5) Hardcoded “precheck” / test prompt behavior
- **File**: `scirag/llm_client.py`
- **Hardcoded content**:
  - A test message: `Reply with the single word: OK`
  - A “model” literal: `model = "precheck-model"`
- **Meaning / risk**:
  - The client may perform a connectivity/credential preflight using this fixed prompt/model name.
  - If the provider enforces model naming strictly, this can fail.

### 6) Hardcoded sentinel values to decide whether an API key “exists”
- **File**: `scirag/llm_client.py`
- **Hardcoded sentinel tuple**:
  - checks like: `self.api_key not in ("lm-studio", "ollama", "local", "not-needed")`
- **Meaning / risk**:
  - These string tokens are effectively part of the “configuration protocol”.
  - If someone sets `PRIMARY_API_KEY="local"` by accident, the code will omit Authorization headers.

### 7) Hardcoded server defaults
- **File**: `scirag/api.py`
- **Hardcoded value**:
  - `uvicorn.run(..., host="127.0.0.1", port=8000, ...)`
- **Meaning / risk**:
  - When serving the API, it binds only to localhost and fixes port to 8000 unless overridden elsewhere.

### 8) Hardcoded LLM model defaults inside orchestration/components
- **Files** (from scan hits):
  - `scirag/attribution.py` (example shows `model="kimi-latest"`)
  - `scirag/iterative_synthesizer.py` (uses `self.config.kimi_model_outline`)
  - `scirag/verifier.py` (uses `temperature=0.1`)
  - `scirag/symbolic_reasoning.py` (uses `self.config.kimi_model_outline`, `temperature=0.1`)
- **Meaning / risk**:
  - Some components hardcode provider-specific default model names (e.g., `kimi-latest`), while others rely on config.
  - Mixed strategy means developers must verify which parts are config-driven vs literal strings.

## Non-Obvious Behavior: Why this is “Hardcoded” even if env vars exist
- The code frequently uses **env-backed defaults** in `scirag/config.py`, but it also includes **literal fallback values** inside clients/managers (`local_llm_manager`, `embedder`, `llm_client` prechecks).
- That means “setting env vars” may not fully control runtime behavior if the fallback path is triggered (e.g., missing/invalid primary credentials → local fallback).

## Suggested Checklist for a Developer
- [x] Identify absolute paths (e.g., `/tmp/...`) that break on Windows/containers.
- [x] Identify default storage locations for indexes/chunks (`./scirag_data/...`).
- [ ] Verify whether config/env can override **all** fallback URLs/models.
- [ ] Confirm whether the LLM precheck (`precheck-model`, fixed “OK” prompt) is configurable.
- [ ] Check whether Uvicorn host/port are configurable when running the server.
- [ ] Document the “sentinel” API key values (`local`, `not-needed`, etc.) in the README.

## Suggested Reading Order (hardcoded hotspots first)
1. `scirag/embedder.py` — local model path + embedding dimension
2. `scirag/config.py` — default paths and default base URLs/models
3. `scirag/llm_client.py` — provider routing, precheck prompt/model, API key sentinel logic
4. `scirag/local_llm_manager.py` — localhost base URL + default model
5. `scirag/api.py` — uvicorn host/port defaults

## Notes / Scope Limits
This report is based on targeted regex search hits in `*.py` and inspection snippets returned by the search tool; it captures the **most consequential** embedded defaults and literals, but there may be additional hardcoded values not matched by the scan patterns.
