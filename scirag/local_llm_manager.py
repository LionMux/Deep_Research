"""Auto-start and manage a local LLM server (LM Studio / Ollama) for fallback."""

import atexit
import logging
import subprocess
import time
from typing import Optional

logger = logging.getLogger(__name__)


class LocalLLMManager:
    """
    Manages the lifecycle of a local LLM server.

    - Detects whether LM Studio or Ollama is installed.
    - If the configured fallback endpoint is down, tries to start the server.
    - Ensures the requested model is loaded.
    - Shuts down the server on process exit *only* if SciRAG started it.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "qwen2.5-3b-instruct",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._started_by_us = False
        self._backend: Optional[str] = None  # "lmstudio" | "ollama"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_running(self, timeout_sec: int = 60) -> bool:
        """Make sure the local server is up and the model is loaded."""
        if self._is_responsive():
            logger.info(f"Local LLM server already responsive at {self.base_url}")
            return self._ensure_model_loaded()

        # Not running — try to auto-start
        if self._try_start():
            self._started_by_us = True
            atexit.register(self.shutdown)
            if self._wait_for_ready(timeout_sec):
                return self._ensure_model_loaded()
            else:
                logger.error("Local server did not become ready in time")
                return False

        logger.error(
            "No local LLM server is running and none of the supported "
            "CLIs (lms, ollama) could be found or started."
        )
        return False

    def shutdown(self):
        """Stop the server **only** if SciRAG started it."""
        if not self._started_by_us:
            return

        logger.info("Shutting down local LLM server (started by SciRAG)")
        try:
            if self._backend == "lmstudio":
                subprocess.run(
                    ["lms", "server", "stop"],
                    check=False,
                    capture_output=True,
                    timeout=30,
                )
            elif self._backend == "ollama":
                # ollama serve runs in foreground; we would need to kill the process.
                # For now just log a warning.
                logger.warning(
                    "Ollama was started by SciRAG but automatic shutdown is not "
                    "implemented. Please stop it manually if desired."
                )
        except Exception as e:
            logger.warning(f"Error while shutting down local server: {e}")
        finally:
            self._started_by_us = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_responsive(self) -> bool:
        """HTTP health-check against the local OpenAI-compatible endpoint."""
        try:
            import requests

            r = requests.get(f"{self.base_url}/models", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def _wait_for_ready(self, timeout_sec: int) -> bool:
        """Poll the endpoint until it responds or we time out."""
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self._is_responsive():
                logger.info("Local LLM server is ready")
                return True
            time.sleep(1)
        return False

    def _try_start(self) -> bool:
        """Attempt to start LM Studio or Ollama."""
        # 1. LM Studio
        if self._has_cli("lms"):
            logger.info("Attempting to start LM Studio server...")
            try:
                # Windows: create a new process group so the server survives
                # our process if we crash (we will shut it down gracefully in
                # shutdown() when possible).
                import sys

                kwargs = {}
                if sys.platform == "win32":
                    kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

                proc = subprocess.Popen(
                    ["lms", "server", "start"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    **kwargs,
                )
                # We don't wait() here — the server is a daemon.
                self._backend = "lmstudio"
                logger.info(f"LM Studio server start initiated (pid={proc.pid})")
                return True
            except Exception as e:
                logger.warning(f"Failed to start LM Studio server: {e}")

        # 2. Ollama
        if self._has_cli("ollama"):
            logger.info("Attempting to start Ollama server...")
            try:
                import sys

                kwargs = {}
                if sys.platform == "win32":
                    kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

                proc = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    **kwargs,
                )
                self._backend = "ollama"
                logger.info(f"Ollama server start initiated (pid={proc.pid})")
                return True
            except Exception as e:
                logger.warning(f"Failed to start Ollama server: {e}")

        return False

    def _ensure_model_loaded(self) -> bool:
        """Make sure the target model is present in memory."""
        if self._backend == "lmstudio":
            return self._ensure_lmstudio_model()
        if self._backend == "ollama":
            return self._ensure_ollama_model()
        # Unknown backend — assume the user handles model loading manually
        return True

    def _ensure_lmstudio_model(self) -> bool:
        """Use `lms ps` and `lms load` to ensure the model is loaded."""
        try:
            result = subprocess.run(
                ["lms", "ps"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if self.model in result.stdout:
                logger.info(f"Model '{self.model}' is already loaded in LM Studio")
                return True
        except Exception:
            pass

        logger.info(f"Loading model '{self.model}' into LM Studio...")
        try:
            result = subprocess.run(
                ["lms", "load", self.model, "-y"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                logger.info(f"Model '{self.model}' loaded successfully")
                return True
            else:
                logger.warning(f"lms load failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Exception while loading model: {e}")
            return False

    def _ensure_ollama_model(self) -> bool:
        """Pull / run the model via Ollama CLI."""
        try:
            # Check if model exists locally
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if self.model in result.stdout:
                logger.info(f"Ollama model '{self.model}' is available")
                return True

            # Try to pull
            logger.info(f"Pulling Ollama model '{self.model}'...")
            pull = subprocess.run(
                ["ollama", "pull", self.model],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if pull.returncode == 0:
                logger.info(f"Ollama model '{self.model}' pulled successfully")
                return True
            else:
                logger.warning(f"ollama pull failed: {pull.stderr}")
                return False
        except Exception as e:
            logger.error(f"Exception with Ollama: {e}")
            return False

    @staticmethod
    def _has_cli(name: str) -> bool:
        """Check whether a CLI tool is on PATH."""
        try:
            subprocess.run(
                [name, "--version"],
                check=False,
                capture_output=True,
                timeout=5,
            )
            return True
        except Exception:
            return False
