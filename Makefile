PY ?= .venv/bin/python
VENV ?= .venv

.PHONY: setup run test cov lint format typecheck evals audit clean

setup:            ## create venv, install pinned deps + Chromium
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.lock -r requirements-dev.txt
	$(PY) -m playwright install chromium
	@test -f .env || (cp .env.example .env && echo "Created .env; add your LLM_API_KEY")

run:              ## start simulated world (:8001) + AutoWork UI (:8000)
	$(PY) run.py

test:             ## offline test suite (no API key needed)
	$(PY) -m pytest -q

cov:              ## tests with coverage report
	$(PY) -m pytest -q --cov --cov-report=term-missing

lint:             ## ruff lint + format check + mypy
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
	$(PY) -m mypy

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

typecheck:
	$(PY) -m mypy

evals:            ## live eval suite (needs LLM_API_KEY); ARGS="--only acme_invoice --repeat 3"
	$(PY) -m evals.run_evals $(ARGS)

audit:            ## known-vulnerability scan of installed packages
	$(VENV)/bin/pip install -q pip-audit && $(VENV)/bin/pip-audit

clean:
	rm -rf runs data workspace .pytest_cache .mypy_cache .ruff_cache .coverage
