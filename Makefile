.PHONY: help install test lint typecheck check build publish clean infbench trained-bench

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
	@echo "  make infbench       Run full inference benchmark suite (sweep + 21 plots)"
	@echo "  make trained-bench  Run trained-model benchmark pipeline (analysis + batch sweep)"
	@echo "  make build      Build sdist + wheel into dist/"
	@echo "  make publish    Publish dist/ to PyPI (requires twine + credentials)"
	@echo "  make clean      Remove build artefacts"

install:
	pip install --user "jax[cpu]" jaxlib
	pip install --user -e ".[all]"

test:
	JAX_PLATFORM_NAME=cpu $(PYTHON) -m pytest tests/ --ignore=tests/test_sweep_simulation.py -v --tb=short \
		--cov=$(PACKAGE) --cov=core --cov-report=term-missing --cov-report=html:docs/coverage

lint:
	$(PYTHON) -m ruff check $(PACKAGE)/ core/ utils/

typecheck:
	$(PYTHON) -m mypy $(PACKAGE)/ core/

check: lint typecheck test

build:
	$(PYTHON) -m build

publish: build
	twine check dist/*
	twine upload dist/*

infbench:
	$(PYTHON) benchmarks/run_all.py

trained-bench:
	$(PYTHON) benchmarks/run_all.py --trained --inference-off

clean:
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
