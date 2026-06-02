#!/usr/bin/env python3
"""Compatibility shim for the local-LLM orchestrator.

The orchestration loop is now the ``orchestrate`` subcommand of the unified
``scirag`` CLI (:mod:`scirag.cli`). ``python cli_orchestrator.py "<query>"``
forwards to ``scirag orchestrate "<query>"`` so existing invocations keep
working:

    python cli_orchestrator.py "Your research question" --index ./samples/papers

Prefer the console script: ``scirag orchestrate "Your research question"``.
"""

import sys

from scirag.cli import main

if __name__ == "__main__":
    # Inject the subcommand so the legacy positional/flags map onto `orchestrate`.
    sys.argv.insert(1, "orchestrate")
    sys.exit(main())
