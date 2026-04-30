.PHONY: help install test lint typecheck check build publish clean

PYTHON  ?= python
PACKAGE  = dantinox

help:
	@echo "DantinoX development targets"
	@echo ""
	@echo "  make install    Install package in editable mode with all dev deps"
	@echo "  make test       Run the test suite"
	@echo "  make lint       Lint with ruff"
	@echo "  make typecheck  Type-check with mypy"
	@echo "  make check      lint + typecheck + test (run before every push)"
	@echo "  make build      Build sdist + wheel into dist/"
	@echo "  make publish    Publish dist/ to PyPI (requires twine + credentials)"
	@echo "  make clean      Remove build artefacts"

install:
	pip install --user "jax[cpu]" jaxlib
	pip install --user -e ".[all]"

test:
	JAX_PLATFORM_NAME=cpu pytest tests/ --ignore=tests/test_sweep_simulation.py -v --tb=short

lint:
	ruff check $(PACKAGE)/ core/ utils/

typecheck:
	mypy $(PACKAGE)/ core/

check: lint typecheck test

build:
	$(PYTHON) -m build

publish: build
	twine check dist/*
	twine upload dist/*

clean:
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
