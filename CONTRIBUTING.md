# Contributing to SciRAG

Thanks for contributing! This guide covers the local development workflow.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,api,mcp]"   # or: pip install -r requirements.txt
pre-commit install                 # enable git hooks
```

Copy `.env.example` to `.env` and fill in the provider keys you need.

## Workflow

- **Format & lint:** `make format` (auto-fix) and `make lint` (`ruff check .`).
- **Test:** `make test` (runs `pytest`). All tests must pass before opening a PR.
- **Pre-commit:** hooks run `ruff` and basic hygiene checks on staged files.
  Run on everything with `make precommit`.

CI runs `ruff check .` and `pytest` on Python 3.10 and 3.12 for every push and
pull request. Keep CI green.

## Conventions

- Code under `third_party/` is vendored from upstream projects — do not lint,
  format, or hand-edit it.
- Keep changes focused; avoid repo-wide reformatting in feature PRs.
- Do not commit large data, model weights, caches, or `.env` files.
- Place imports at the top of modules unless a lazy/conditional import is
  required (those are intentionally allowed via the `E402` ruff exception).
