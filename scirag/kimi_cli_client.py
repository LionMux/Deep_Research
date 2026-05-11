"""Kimi CLI wrapper — uses local `kimi` binary instead of HTTP API."""

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class KimiCLIClient:
    """Client that delegates to the local `kimi` CLI binary."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("KIMI_API_KEY", "")
        self.model = model or os.getenv("KIMI_MODEL_CODE", "")
        self._kimi_path = self._find_kimi_binary()
        if not self._kimi_path:
            raise RuntimeError("kimi CLI not found.")
        logger.info(f"KimiCLIClient using binary: {self._kimi_path}")

    def chat(self, messages: List[Dict], model: str = "", temperature: float = 0.3, max_tokens: int = 4096, response_format: Optional[dict] = None) -> str:
        prompt = self._messages_to_prompt(messages)
        m = model or self.model
        return self._run_kimi(prompt, model=m)

    def chat_json(self, messages: List[Dict], model: str = "", temperature: float = 0.3, max_tokens: int = 4096) -> dict:
        content = self.chat(messages, model, temperature, max_tokens)
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON: {content[:200]}")
            return {"raw": content}

    def _find_kimi_binary(self) -> Optional[str]:
        binary_name = "kimi.exe" if sys.platform == "win32" else "kimi"
        for path_dir in os.environ.get("PATH", "").split(os.pathsep):
            candidate = os.path.join(path_dir, binary_name)
            if os.path.isfile(candidate):
                return candidate
        candidates = [os.path.expanduser(r"~\.local\bin\kimi.exe"), os.path.expanduser("~/.local/bin/kimi")]
        for c in candidates:
            if os.path.isfile(c):
                return c
        return None

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[System]\n{content}")
            elif role == "user":
                parts.append(f"[User]\n{content}")
            elif role == "assistant":
                parts.append(f"[Assistant]\n{content}")
            else:
                parts.append(f"[{role}]\n{content}")
        return "\n\n".join(parts)

    def _run_kimi(self, prompt: str, model: str = "") -> str:
        work_dir = tempfile.mkdtemp(prefix="kimi_cli_")
        env = os.environ.copy()
        for key in list(env.keys()):
            if key.startswith("KIMI_"):
                env.pop(key, None)
        try:
            cmd = [self._kimi_path, "--work-dir", work_dir, "--prompt", prompt, "--print", "--final-message-only"]
            if model:
                cmd.extend(["--model", model])
            print(f"[KimiCLI] CMD: {cmd}")
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=work_dir, timeout=300)
            print(f"[KimiCLI] RC={result.returncode} stdout={repr(result.stdout)} stderr={repr(result.stderr)}")
            if result.returncode != 0 and not result.stdout.strip():
                raise RuntimeError(f"kimi CLI failed (rc={result.returncode}): {result.stderr[:500]}")
            text = result.stdout.strip()
            text = re.sub(r"<choice>.*?</choice>", "", text, flags=re.S).strip()
            return text
        finally:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)
