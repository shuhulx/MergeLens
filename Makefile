.DEFAULT_GOAL := help

.PHONY: help install format lint test coverage build clean ci

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install for development
	python -m pip install -e ".[dev,all]"

format:  ## Auto-format code
	ruff format .
	ruff check --fix .

lint:  ## Run linters
	ruff check .
	ruff format --check .
	mypy src/mergelens

test:  ## Run tests
	python -m pytest -q

coverage:  ## Run tests with coverage
	python -m pytest --cov=mergelens --cov-report=term-missing tests/

build:  ## Build wheel and source distribution
	python -m build

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info src/*.egg-info .ruff_cache/ .mypy_cache/ .pytest_cache/ htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +

ci: lint test build  ## Run local static, test, and package checks
