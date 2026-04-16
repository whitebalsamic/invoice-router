PYTHON ?= python3.12
PIP ?= $(PYTHON) -m pip

.PHONY: install lint format typecheck test unit integration e2e build up down demo

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

typecheck:
	mypy

test:
	pytest -q

unit:
	pytest -m "not integration and not e2e"

integration:
	pytest -m integration

e2e:
	pytest -m e2e

build:
	$(PYTHON) -m build

up:
	docker compose up -d postgres redis

down:
	docker compose down

demo:
	invoice-router demo --workspace .demo
