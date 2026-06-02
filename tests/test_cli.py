"""Smoke tests for the consolidated `scirag` CLI.

Only offline subcommands are exercised (eval / help); networked subcommands
(query/api/mcp/orchestrate) are validated for registration, not executed.
"""

import json
from pathlib import Path

import pytest

from scirag import cli

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = REPO_ROOT / "benchmarks" / "retrieval_benchmark.json"
CORPUS = REPO_ROOT / "samples" / "papers"


def test_eval_subcommand_runs(monkeypatch, capsys):
    argv = [
        "scirag", "eval",
        "--benchmark", str(BENCHMARK),
        "--corpus", str(CORPUS),
        "--mode", "sparse",
        "--top-k", "3",
    ]
    monkeypatch.setattr("sys.argv", argv)
    rc = cli.main()
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["mode"] == "sparse"
    assert data["recall@3"] == 1.0


def test_help_no_command_returns_1(monkeypatch):
    monkeypatch.setattr("sys.argv", ["scirag"])
    assert cli.main() == 1


@pytest.mark.parametrize(
    "command",
    ["query", "index", "search", "synthesize", "extract", "verify",
     "orchestrate", "eval", "api", "mcp", "status", "install-claude"],
)
def test_subcommands_registered(monkeypatch, command):
    # Each subcommand must parse `--help` without error (SystemExit(0)).
    monkeypatch.setattr("sys.argv", ["scirag", command, "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0


def test_main_shim_delegates():
    import main as main_shim
    assert main_shim.main is cli.main
