.PHONY: help install test lint typecheck doccheck check docs-build docs-serve build publish bump-patch bump-minor bump-major clean infbench trained-bench diffbench diffusion-train diffusion-train-dry benchmark-full

PYTHON  ?= python
PACKAGE  = dantinox

help:
	@echo "DantinoX development targets"
	@echo ""
	@echo "  ── Quality ────────────────────────────────────────────────────────"
	@echo "  make install       Install in editable mode with all dev deps"
	@echo "  make test          Run the test suite (CPU JAX)"
	@echo "  make lint          Lint with ruff"
	@echo "  make typecheck     Type-check with mypy"
	@echo "  make doccheck      Docstring coverage with interrogate (100 % required)"
	@echo "  make check         lint + typecheck + test + doccheck (run before every push)"
	@echo ""
	@echo "  ── Documentation ──────────────────────────────────────────────────"
	@echo "  make docs-serve    Start local MkDocs dev server at http://127.0.0.1:8000"
	@echo "  make docs-build    Build static docs into site/"
	@echo ""
	@echo "  ── Benchmarks ─────────────────────────────────────────────────────"
	@echo "  make infbench      Full AR inference benchmark suite (sweep + 21 plots)"
	@echo "  make diffbench     AR vs Diffusion + Fast-dLLM benchmark suite"
	@echo "  make trained-bench Trained-model benchmark pipeline (analysis + batch sweep)"
	@echo ""
	@echo "  ── Release ────────────────────────────────────────────────────────"
	@echo "  make bump-patch    Bump version x.y.Z → x.y.(Z+1)"
	@echo "  make bump-minor    Bump version x.Y.z → x.(Y+1).0"
	@echo "  make bump-major    Bump version X.y.z → (X+1).0.0"
	@echo "  make build         Build sdist + wheel into dist/"
	@echo "  make publish       Publish dist/ to PyPI (requires twine + credentials)"
	@echo "  make clean         Remove build artefacts"

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

doccheck:
	$(PYTHON) -m interrogate \
		--verbose \
		--fail-under=100 \
		--ignore-init-module \
		--ignore-magic \
		--ignore-private \
		--ignore-semiprivate \
		$(PACKAGE)/ core/

check: lint typecheck doccheck test

# ── Documentation ─────────────────────────────────────────────────────────────
docs-serve:
	PYTHONPATH=. mkdocs serve

docs-build:
	PYTHONPATH=. mkdocs build --strict

bump-patch:
	$(PYTHON) -c "import re,pathlib; p=pathlib.Path('pyproject.toml'); t=p.read_text(); v=re.search(r'version = \"(\d+)\.(\d+)\.(\d+)\"',t); a,b,c=int(v.group(1)),int(v.group(2)),int(v.group(3)); nv=f'{a}.{b}.{c+1}'; p.write_text(t.replace(v.group(0),f'version = \"{nv}\"')); print('Bumped to',nv)"

bump-minor:
	$(PYTHON) -c "import re,pathlib; p=pathlib.Path('pyproject.toml'); t=p.read_text(); v=re.search(r'version = \"(\d+)\.(\d+)\.(\d+)\"',t); a,b,c=int(v.group(1)),int(v.group(2)),int(v.group(3)); nv=f'{a}.{b+1}.0'; p.write_text(t.replace(v.group(0),f'version = \"{nv}\"')); print('Bumped to',nv)"

bump-major:
	$(PYTHON) -c "import re,pathlib; p=pathlib.Path('pyproject.toml'); t=p.read_text(); v=re.search(r'version = \"(\d+)\.(\d+)\.(\d+)\"',t); a,b,c=int(v.group(1)),int(v.group(2)),int(v.group(3)); nv=f'{a+1}.0.0'; p.write_text(t.replace(v.group(0),f'version = \"{nv}\"')); print('Bumped to',nv)"

build:
	$(PYTHON) -m build

publish: build
	twine check dist/*
	twine upload dist/*

infbench:
	$(PYTHON) benchmarks/run_all.py

# ── AR vs Diffusion + Fast-dLLM benchmark ─────────────────────────────────────
DIFF_OUT    ?= results/diffusion_ar_sweep.csv
DIFF_PLOTS  ?= plots/diffusion_ar

diffbench: $(DIFF_OUT)
	$(PYTHON) benchmarks/plot_diffusion_ar.py --csv $(DIFF_OUT) --out $(DIFF_PLOTS)
	@echo "Plots saved → $(DIFF_PLOTS)/"

$(DIFF_OUT):
	mkdir -p results
	$(PYTHON) benchmarks/diffusion_ar_sweep.py --out $(DIFF_OUT)

trained-bench:
	$(PYTHON) benchmarks/run_all.py --trained --inference-off

clean:
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

# ── Diffusion training ────────────────────────────────────────────────────────
diffusion-train:
	bash scripts/train_diffusion_suite.sh

diffusion-train-dry:
	bash scripts/train_diffusion_suite.sh --dry-run

ar-train:
	bash scripts/train_ar_suite.sh

ar-train-dry:
	bash scripts/train_ar_suite.sh --dry-run

# ── Full EMNLP pipeline (training + benchmarks + figures) ─────────────────────
emnlp-full:
	bash scripts/run_full_emnlp.sh

emnlp-benchmarks-only:
	bash scripts/run_full_emnlp.sh --skip-training

emnlp-plots-only:
	bash scripts/run_full_emnlp.sh --only-plots

# ── cleanup duplicate runs to free disk space ─────────────────────────────────
cleanup-runs-dry:
	$(PYTHON) scripts/cleanup_runs.py --runs-dir runs

cleanup-runs:
	$(PYTHON) scripts/cleanup_runs.py --runs-dir runs --execute

# ── Full EMNLP benchmark pipeline (run_all.py wrapper) ───────────────────────
benchmark-full:
	$(PYTHON) benchmarks/run_all.py --diff-ar --eval --pdf --verbose
