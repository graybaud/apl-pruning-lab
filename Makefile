.PHONY: help install test test-unit test-int clean clean-all coverage

help:
	@echo "APL Pruning Lab — Available commands:"
	@echo ""
	@echo "  make install       Install in development mode"
	@echo "  make test          Run all tests"
	@echo "  make test-unit     Run unit tests only"
	@echo "  make test-int      Run integration tests only"
	@echo "  make coverage      Run tests with coverage report"
	@echo "  make clean         Remove cache files"
	@echo "  make clean-all     Remove cache + venv + eggs"
	@echo ""

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-unit:
	pytest tests/ -v -m "not integration"

test-int:
	pytest tests/ -v -m "integration"

coverage:
	pytest tests/ --cov=. --cov-report=term-missing --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

clean:
	@echo "Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Done."

clean-all: clean
	@echo "Removing venv..."
	rm -rf venv/ venv-apl/
	@echo "Done."
