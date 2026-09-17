SHELL := /bin/bash
GIT_BRANCH := $(shell git branch --show-current)
PY_VENV := .venv/
UV_LOCKFILE := uv.lock

#------------------------------------------------------------------------------
# Default help target
#------------------------------------------------------------------------------

help:
	@echo "Available targets:"
	@awk -F':' '/^[a-zA-Z0-9\._-]+:/ && !/^[ \t]*\.PHONY/ {print $$1}' $(MAKEFILE_LIST) | sort -u | column


#------------------------------------------------------------------------------
# Bootstrap project with uv
#------------------------------------------------------------------------------

$(UV_LOCKFILE):
	uv lock --build-isolation

$(PY_VENV): $(UV_LOCKFILE)
	uv sync --frozen

.PHONY: clean
clean:
	rm -rf $(PY_VENV)
	rm -f tissage_cosmique.db
	find src -type d -name '__pycache__' | xargs rm -rf
	find tests -type d -name '__pycache__' | xargs rm -rf

.PHONY: init
init: $(PY_VENV)
	uv run pre-commit install

.PHONY: update-deps
update-deps: init
	uv lock --upgrade --build-isolation

.PHONY: update
update: update-deps init


#------------------------------------------------------------------------------
# Lint and type checking
#------------------------------------------------------------------------------

.PHONY: lint
lint:
	pre-commit run --all-files

.PHONY: typing
typing:
	mypy src tests


#------------------------------------------------------------------------------
# Testing with SQLite
#------------------------------------------------------------------------------

.PHONY: test-sqlite
test-sqlite: export TISSAGE_COSMIQUE__DB__URL=sqlite+aiosqlite:////${PWD}/tests/test_tissage_cosmique.db
test-sqlite:
	pytest -vvv --asyncio-mode=auto --cov=tissage_cosmique --cov-branch --cov-report=term --cov-report=html ${PYTEST_ARGS}

.PHONY: test-github-ci
test-github-ci: export TISSAGE_COSMIQUE__DB__URL=sqlite+aiosqlite:////${PWD}/tests/test_tissage_cosmique.db
test-github-ci:
	pytest -vvv --asyncio-mode=auto --cov=tissage_cosmique --cov-branch --cov-report=term --cov-report=xml ${PYTEST_ARGS}
