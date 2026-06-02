#!/usr/bin/env python3
"""Compatibility shim for the unified SciRAG CLI.

The CLI now lives in :mod:`scirag.cli` (installed as the ``scirag`` console
script). ``python main.py <args>`` simply forwards to it, so all existing
invocations keep working:

    python main.py query "..." --papers-dir ./samples/papers
    python main.py index ./samples/papers
    python main.py search "..." --top-k 5
    python main.py api --port 8000
    python main.py mcp

Prefer the ``scirag`` console script (e.g. ``scirag query "..."``).
"""

import sys

from scirag.cli import main

if __name__ == "__main__":
    sys.exit(main())
