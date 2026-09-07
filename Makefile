PYTHON ?= python3
VENV ?= venv
# Windows venvs use Scripts/ instead of bin/
ifeq ($(OS),Windows_NT)
    BIN := Scripts
else
    BIN := bin
endif
PIP = $(VENV)/$(BIN)/pip
PY = $(VENV)/$(BIN)/python

.PHONY: help setup install test build run clean

help:
	@echo "Targets:"
	@echo "  setup   create virtualenv ($(VENV))"
	@echo "  install install deps + package into $(VENV)"
	@echo "  test    run unit tests (unittest discover)"
	@echo "  build   build sdist + wheel (requires 'build')"
	@echo "  run     launch the interactive TUI agent"
	@echo "  clean   remove virtualenv and build artifacts"

setup: $(VENV)/$(BIN)/python

$(VENV)/$(BIN)/python:
	$(PYTHON) -m venv $(VENV)

install: setup
	$(PIP) install rich httpx prompt_toolkit
	$(PIP) install -e .

test: install
	$(PY) -m unittest discover -s tests -v

build: setup
	$(PIP) install build
	$(PY) -m build

run: install
	$(PY) -m python_agent_harness.cli run

clean:
	rm -rf $(VENV) build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
