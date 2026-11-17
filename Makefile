.PHONY: install run ingest test lint format docker

install:
	pip install -e ".[local,dev]"

run:
	uvicorn rag_engine.main:app --reload --port 8000

ingest:
	python -m rag_engine ingest ./examples/docs --collection demo --init-db

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .

docker:
	docker compose up --build
