PYTHONPATH := src
export PYTHONPATH

.PHONY: install ingest run export test test-unit test-integration lint

# Install all dependencies (no editable install needed — PYTHONPATH=src handles resolution)
install:
	pip3 install -r requirements.txt

# Populate Pinecone with transcript chunks (run once before the agent)
# Usage: make ingest TICKERS="AMZN GOOG META"
TICKERS ?= AMZN GOOG META
ingest:
	python3 scripts/ingest_transcripts.py --tickers $(TICKERS)

# Run the agent interactively for a single ticker
# Usage: make run TICKER=AMZN
TICKER ?= AMZN
run:
	python3 scripts/run_agent.py --ticker $(TICKER)

# Export the feedback store to CSV for analysis
export:
	python3 scripts/export_feedback.py

# Unit tests only (no API calls)
test-unit:
	python3 -m pytest tests/unit -v

# Integration tests (require live API keys)
test-integration:
	python3 -m pytest tests/integration -v -m integration

# All tests
test: test-unit

lint:
	python3 -m pylint src/earnings_research_agent --fail-under=8

install-backend:
	pip3 install -r backend/requirements.txt

install-frontend:
	cd frontend && npm install

server:
	PYTHONPATH=src python3 -m uvicorn backend.main:app --reload --port 8000

frontend-dev:
	cd frontend && npm run dev

dev: server  # hint: run 'make server' and 'make frontend-dev' in separate terminals
