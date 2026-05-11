"""SciRAG LLM client.

Goal for this project run:
- Avoid PRIMARY (ai.wormsoft.ru) when HF is configured.
- Keep outline/verification working for MCP deep-research full_report by routing chat() through HF first.

Routing order for chat():
1) HF (chat via HF Inference API) when HF_API_TOKEN + HF_MODEL_SYNTHESIS + HF_BASE_URL are configured.
2) PRIMARY (OpenAI-compatible) chain fallback: primary → Gemini → local (optional).

Note: chat_synthesize() is HF-only -> Gemini fallback (kept for backward compatibility).
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PrimaryClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        fallback_url: Optional[str] = None,
        fallback_model: Optional[str] = None,
        gemini_api_key: str = "",
        gemini_model: str = "",
        gemini_base_url: str = "",
    ):
        self.api_key = api_key or ""
        self.primary_url = (base_url or "").rstrip("/")
        self.fallback_url = fallback_url.rstrip("/") if fallback_url else None
        self.fallback_model = fallback_model

        self.gemini_api_key = gemini_api_key or ""
        self.gemini_model = gemini_model or ""
        self.gemini_base_url = gemini_base_url.rstrip("/") if gemini_base_url else ""

        self.base_url = self.primary_url
        self._use_fallback = False

        # OpenAI SDK (optional)
        self._client = None
        try:
            from openai import OpenAI  # type: ignore

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            logger.info("PrimaryClient: using openai SDK for primary provider")
        except Exception:
            logger.info("PrimaryClient: openai SDK not available; using raw HTTP for primary provider")

        # HF config (set by from_config)
        self.hf_api_token: str = ""
        self.hf_base_url: str = ""
        self.hf_model_synthesis: str = ""
        self.hf_precheck_enabled: bool = True
        self.hf_precheck_timeout_sec: int = 20
        self._hf_precheck_done: bool = False
        self._hf_precheck_ok: bool = False

    def chat_synthesize(
        self,
        messages: List[Dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """
        Backward-compatible alias used by IterativeSynthesizer.

        Historically this was HF-only; now it simply routes through the same
        `chat()` method, preserving "HF first, never touch PRIMARY when HF is configured".
        """
        # `model` is derived inside chat() from HF configuration; keep a stable default.
        return self.chat(
            messages=messages,
            model=getattr(self, "hf_model_synthesis", "") or "synthesis-model",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @classmethod
    def from_config(cls, config: Any) -> "PrimaryClient":
        client = cls(
            api_key=getattr(config, "primary_api_key", "") or getattr(config, "kimi_api_key", ""),
            base_url=getattr(config, "primary_base_url", "") or getattr(config, "kimi_base_url", ""),
            fallback_url=getattr(config, "primary_fallback_url", None) or getattr(config, "kimi_fallback_url", None),
            fallback_model=getattr(config, "primary_fallback_model", None) or getattr(config, "kimi_fallback_model", None),
            gemini_api_key=getattr(config, "gemini_api_key", "") or "",
            gemini_model=getattr(config, "gemini_model", "") or "",
            gemini_base_url=getattr(config, "gemini_base_url", "") or "",
        )

        # HF settings
        client.hf_api_token = getattr(config, "hf_api_token", "") or ""
        client.hf_base_url = getattr(config, "hf_base_url", "") or ""
        client.hf_model_synthesis = getattr(config, "hf_model_synthesis", "") or ""
        client.hf_precheck_enabled = bool(getattr(config, "hf_precheck_enabled", True))
        client.hf_precheck_timeout_sec = int(getattr(config, "hf_precheck_timeout_sec", 20))
        return client

    def _has_real_key(self) -> bool:
        return bool(self.api_key and self.api_key not in ("lm-studio", "ollama", "local", "not-needed"))

    def _activate_fallback(self) -> None:
        self._use_fallback = True
        if not self.fallback_url:
            raise RuntimeError("Local fallback is requested but fallback_url is not configured.")
        self.base_url = self.fallback_url

        # Auto-start local server if it is not already running
        try:
            from scirag.local_llm_manager import LocalLLMManager  # local import

            manager = LocalLLMManager(
                base_url=self.fallback_url,
                model=self.fallback_model or "qwen2.5-3b-instruct",
            )
            manager.ensure_running()
        except Exception as e:
            logger.warning(f"Local LLM auto-start failed: {e}")

        # If OpenAI SDK is in use, rewire it to fallback base_url
        if self._client is not None:
            try:
                from openai import OpenAI  # type: ignore

                self._client = OpenAI(api_key="not-needed", base_url=self.fallback_url)
            except Exception:
                self._client = None

    # ---------------- HF ----------------
    def _hf_configured(self) -> bool:
        return bool(self.hf_api_token and self.hf_base_url and self.hf_model_synthesis)

    # ---------------- DeepInfra ----------------
    def _deepinfra_configured(self) -> bool:
        return bool(
            getattr(self, "deepinfra_api_token", "") and
            getattr(self, "deepinfra_base_url", "") and
            getattr(self, "deepinfra_model_synthesis", "")
        )

    def _hf_precheck_once(self) -> bool:
        if not self._hf_configured():
            return False
        if self._hf_precheck_done:
            return self._hf_precheck_ok

        if not self.hf_precheck_enabled:
            self._hf_precheck_ok = True
            self._hf_precheck_done = True
            return True

        try:
            self._hf_precheck_ok = self._hf_token_precheck(
                token=self.hf_api_token,
                base_url=self.hf_base_url,
                model=self.hf_model_synthesis,
                timeout_sec=self.hf_precheck_timeout_sec,
            )
        except Exception as e:
            logger.warning(f"HF token precheck failed: {e}")
            self._hf_precheck_ok = False
        finally:
            self._hf_precheck_done = True

        return self._hf_precheck_ok

    @staticmethod
    def _build_prompt_from_messages(messages: List[Dict]) -> str:
        sys_parts: List[str] = []
        convo_parts: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role == "system":
                sys_parts.append(content)
            elif role == "user":
                convo_parts.append(f"[User]\n{content}")
            elif role == "assistant":
                convo_parts.append(f"[Assistant]\n{content}")
            else:
                convo_parts.append(f"[{role}]\n{content}")

        prompt_parts: List[str] = []
        if sys_parts:
            prompt_parts.append("[System]\n" + "\n\n".join(sys_parts))
        if convo_parts:
            prompt_parts.append("\n".join(convo_parts))
        return "\n\n".join(prompt_parts).strip()

    def _hf_token_precheck(self, *, token: str, base_url: str, model: str, timeout_sec: int) -> bool:
        """
        Best-effort HF readiness check.

        Supports both:
        - legacy HF Inference API style (base_url like https://api-inference.huggingface.co/models)
        - Inference Providers router style (base_url like https://router.huggingface.co/v1) using /chat/completions
        """
        if not token or not base_url or not model:
            return False

        import requests

        base = base_url.rstrip("/")

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}

        # If using HF router (/v1), validate via chat/completions (OpenAI-compatible)
        if base.endswith("/v1") and "router.huggingface.co" in base:
            url = base + "/chat/completions"
            payload: Dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                "temperature": 0.0,
                "max_tokens": 8,
                "stream": False,
            }
            r = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
            r.raise_for_status()
            return True

        # Legacy HF Inference API style (may 404 for gated/unsupported models)
        url = f"{base}/{model}"
        payload = {
            "inputs": "Reply with the single word OK.",
            "parameters": {"max_new_tokens": 8, "temperature": 0.0},
        }
        r = requests.post(url, headers={"Authorization": f"Bearer {token}"}, json=payload, timeout=timeout_sec)
        r.raise_for_status()
        _ = r.json()
        return True

    def _chat_hf(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """
        Hugging Face Inference Providers (OpenAI-compatible router).

        Uses:
          POST {hf_base_url}/chat/completions
        where hf_base_url default is https://router.huggingface.co/v1
        """
        if not self._hf_configured():
            raise RuntimeError("HF is not configured.")

        import requests

        headers = {"Content-Type": "application/json"}
        if self.hf_api_token:
            headers["Authorization"] = f"Bearer {self.hf_api_token}"

        payload: Dict[str, Any] = {
            "model": self.hf_model_synthesis,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max(1, int(max_tokens)),
            "stream": False,
        }

        url = self.hf_base_url.rstrip("/") + "/chat/completions"
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()

        # OpenAI shape: { choices: [{ message: { content: "..." } }]}
        try:
            msg = data["choices"][0]["message"]
            return (msg.get("content") or msg.get("reasoning") or "").strip()
        except Exception:
            return json.dumps(data)[:5000]

    # ---------------- DeepInfra ----------------
    def _chat_deepinfra(
        self,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        response_format: Optional[dict] = None,
    ) -> str:
        """
        DeepInfra is OpenAI-compatible for /v1/openai chat.completions.
        We call: {deepinfra_base_url}/chat/completions
        """
        import requests

        if not self._deepinfra_configured():
            raise RuntimeError("DeepInfra is not configured")

        # default: use deepinfra_model_synthesis for synthesis/offline phases
        deepinfra_model = getattr(self, "deepinfra_model_synthesis", "") or ""

        headers = {"Content-Type": "application/json"}
        if self.deepinfra_api_token:
            headers["Authorization"] = f"Bearer {self.deepinfra_api_token}"

        payload: Dict[str, Any] = {
            "model": deepinfra_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max(1, int(max_tokens)),
        }
        if response_format:
            payload["response_format"] = response_format

        url = self.deepinfra_base_url.rstrip("/") + "/chat/completions"
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()

        msg = data["choices"][0]["message"]
        return msg.get("content") or msg.get("reasoning") or ""

    # ---------------- chat public ----------------
    def chat(
        self,
        messages: List[Dict],
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        response_format: Optional[dict] = None,
    ) -> str:
        """chat() used by outline/verify. Prefer HF/DeepInfra to avoid PRIMARY 401."""
        # 1) HF preferred when configured: HF -> (optional DeepInfra) ; never touch PRIMARY.
        if self._hf_configured():
            # HF precheck is best-effort only (may be false-negative).
            try:
                _ = self._hf_precheck_once()
            except Exception:
                pass

            try:
                return self._chat_hf(messages=messages, temperature=temperature, max_tokens=max_tokens)
            except Exception as e_hf:
                logger.warning(f"HF chat() failed; trying DeepInfra (if enabled): {e_hf}")

                if self._deepinfra_configured():
                    return self._chat_deepinfra(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format,
                    )

                # No DeepInfra -> keep HF error semantics
                raise

        # 2) When HF is NOT configured: legacy chain (PRIMARY -> Gemini -> local fallback)
        if not self._use_fallback:
            try:
                return self._chat_primary_chain(
                    url=self.primary_url,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
            except Exception as e:
                if self._should_try_gemini(e):
                    return self._chat_gemini(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format,
                    )
                if self.fallback_url:
                    self._activate_fallback()
                    return self._chat_requests(
                        url=self.fallback_url,
                        model=self.fallback_model or model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format,
                    )
                raise

        if not self.fallback_url:
            raise RuntimeError("Local fallback URL is not configured.")
        return self._chat_requests(
            url=self.fallback_url,
            model=self.fallback_model or model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    def chat_json(
        self,
        messages: List[Dict],
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> dict:
        content = self.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        ).strip()

        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        def _try_parse(s: str) -> Optional[dict]:
            try:
                return json.loads(s)
            except Exception:
                return None

        parsed = _try_parse(content)
        if parsed is not None:
            return parsed

        first_brace = content.find("{")
        last_brace = content.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = content[first_brace : last_brace + 1].strip()
            parsed = _try_parse(candidate)
            if parsed is not None:
                return parsed

        return {"raw": content}

    # ---------------- primary/gemini/local ----------------
    def _chat_primary_chain(
        self,
        url: str,
        model: str,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        response_format: Optional[dict],
    ) -> str:
        # Use OpenAI SDK if available; else raw
        if self._client is not None:
            return self._chat_openai(url, model, messages, temperature, max_tokens, response_format)
        return self._chat_requests(url, model, messages, temperature, max_tokens, response_format)

    def _should_try_gemini(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        if "429" in msg:
            return True
        quota_markers = ["rate limit", "rate-limited", "too many requests", "quota", "exceeded", "limit", "resource exhausted"]
        return any(m in msg for m in quota_markers) or ("401" in msg or "403" in msg)

    def _chat_gemini(
        self,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        response_format: Optional[dict] = None,
    ) -> str:
        if not self.gemini_api_key:
            raise RuntimeError("Gemini API key is not configured")
        if not self.gemini_base_url or not self.gemini_model:
            raise RuntimeError("Gemini base_url/model are not configured")

        import requests

        prompt = self._build_prompt_from_messages(messages)
        if response_format and response_format.get("type") == "json_object":
            prompt += "\n\nReturn ONLY valid JSON. Do not wrap it in Markdown."

        url = (
            f"{self.gemini_base_url}/models/{self.gemini_model}:generateContent"
            f"?key={self.gemini_api_key}"
        )

        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max(1, int(max_tokens))},
        }

        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()

        try:
            return data["candidates"][0]["content"]["parts"][0]["text"] or json.dumps(data)[:5000]
        except Exception:
            return json.dumps(data)[:5000]

    def _chat_openai(
        self,
        url: str,
        model: str,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        response_format: Optional[dict],
    ) -> str:
        if self._client is None:
            return self._chat_requests(url, model, messages, temperature, max_tokens, response_format)

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(**kwargs)  # type: ignore[union-attr]
                msg = resp.choices[0].message
                return msg.content or ""
            except Exception as e:
                last_exc = e
                logger.warning(f"Primary OpenAI SDK attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)
        raise last_exc  # type: ignore[misc]

    def _chat_requests(
        self,
        url: str,
        model: str,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        response_format: Optional[dict],
    ) -> str:
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key not in ("lm-studio", "ollama", "local", "not-needed"):
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        r = requests.post(f"{url}/chat/completions", headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        return msg.get("content") or msg.get("reasoning") or ""

    def precheck_or_raise(self) -> None:
        """
        Best-effort precheck.

        Critical: if HF is configured, do NOT touch PRIMARY at all.
        This prevents 401/429 from PRIMARY from breaking deep-research/MCP flows.
        """
        # If HF is configured, we treat it as the source of truth and never call PRIMARY.
        if self._hf_configured():
            try:
                if self._hf_precheck_once():
                    return
            except Exception:
                # HF precheck failed; still do not touch PRIMARY.
                return

        # HF not configured: legacy behavior may attempt PRIMARY precheck (best-effort).
        try:
            self._precheck_primary()
        except Exception as e:
            msg = str(e).lower()
            if not any(m in msg for m in ["429", "401", "403"]):
                raise

    def _precheck_primary(self) -> None:
        test_messages = [{"role": "user", "content": "Reply with the single word: OK"}]
        model = "precheck-model"
        self.chat(
            messages=test_messages,
            model=model,
            temperature=0.0,
            max_tokens=16,
            response_format=None,
        )


# Backward-compatible alias expected by older code
KimiClient = PrimaryClient
