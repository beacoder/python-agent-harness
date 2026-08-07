PYTHON ?= python3
VENV ?= venv
PIP = $(VENV)/bin/pip

.PHONY: help setup install test build run clean

help:
	@echo "Targets:"
	@echo "  setup   create virtualenv ($(VENV))"
	@echo "  install install deps + package into $(VENV)"
	@echo "  test    run unit tests (unittest discover)"
	@echo "  build   build sdist + wheel (requires 'build')"
	@echo "  run     launch the interactive TUI agent"
	@echo "  clean   remove virtualenv and build artifacts"

setup: $(VENV)/bin/python

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)

install: setup
	$(PIP) install rich httpx prompt_toolkit
	$(PIP) install -e .

test: install
	$(VENV)/bin/python -m unittest discover -s tests -v

build: setup
	$(PIP) install build
	$(VENV)/bin/python -m build

run: install
	$(VENV)/bin/python -m python_agent_harness.cli run

clean:
	rm -rf $(VENV) build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
