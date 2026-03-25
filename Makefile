.PHONY: install dev lint type-check test test-cov format check all up down logs

# Setup
install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install

# Code quality
lint:
	ruff check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

type-check:
	mypy src/

test:
	pytest tests/

test-cov:
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

check: lint type-check test

# Docker
up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

all: format check
