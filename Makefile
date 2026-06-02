.PHONY: help install install-dev lint format test precommit clean

help:
	@echo "SciRAG — common tasks"
	@echo "  make install      Install runtime dependencies (requirements.txt)"
	@echo "  make install-dev  Install package in editable mode with dev/api/mcp extras"
	@echo "  make lint         Run ruff lint checks"
	@echo "  make format       Auto-format with ruff (and apply lint fixes)"
	@echo "  make test         Run the pytest suite"
	@echo "  make precommit    Run all pre-commit hooks on the whole repo"
	@echo "  make clean        Remove caches and build artifacts"

install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev,api,mcp]"

lint:
	ruff check .

format:
	ruff check . --fix
	ruff format .

test:
	pytest

precommit:
	pre-commit run --all-files

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -not -path "./third_party/*" -exec rm -rf {} +
