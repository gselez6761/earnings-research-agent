.PHONY: install ingest run export test test-unit test-integration lint

# Install the package and all dependencies in editable mode
install:
	pip install -e . && pip install -r requirements.txt

# Populate Pinecone with transcript chunks (run once before the agent)
# Usage: make ingest TICKERS="AMZN GOOG META"
TICKERS ?= AMZN GOOG META
ingest:
	python scripts/ingest_transcripts.py --tickers $(TICKERS)

# Run the agent interactively for a single ticker
# Usage: make run TICKER=AMZN
TICKER ?= AMZN
run:
	python scripts/run_agent.py --ticker $(TICKER)

# Export the feedback store to CSV for analysis
export:
	python scripts/export_feedback.py

# Unit tests only (no API calls)
test-unit:
	python -m pytest tests/unit -v

# Integration tests (require live API keys)
test-integration:
	python -m pytest tests/integration -v -m integration

# All tests
test: test-unit

lint:
	python -m pylint src/earnings_research_agent --fail-under=8
