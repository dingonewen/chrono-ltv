.PHONY: install install-dev lint type-check test test-unit test-integration test-behavioral \
        simulate train serve monitor docker-build docker-up clean

PYTHON      := python
PIP         := pip
SRC         := src
TESTS       := tests
SCRIPTS     := scripts

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"
	pre-commit install

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------
lint:
	ruff check $(SRC) $(TESTS) $(SCRIPTS)
	ruff format --check $(SRC) $(TESTS) $(SCRIPTS)

format:
	ruff check --fix $(SRC) $(TESTS) $(SCRIPTS)
	ruff format $(SRC) $(TESTS) $(SCRIPTS)

type-check:
	mypy $(SRC)

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
test:
	pytest $(TESTS) -v

test-unit:
	pytest $(TESTS)/unit -v

test-integration:
	pytest $(TESTS)/integration -v

test-behavioral:
	pytest $(TESTS)/behavioral -v

test-ci:
	pytest $(TESTS) -v --cov=$(SRC)/chrono_ltv --cov-report=xml

# ---------------------------------------------------------------------------
# Pipeline entry points
# ---------------------------------------------------------------------------
simulate:
	$(PYTHON) $(SCRIPTS)/generate_data.py

train:
	$(PYTHON) $(SCRIPTS)/train.py

serve:
	uvicorn src.chrono_ltv.serving.api:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run src/chrono_ltv/dashboard.py

monitor:
	$(PYTHON) $(SCRIPTS)/monitor.py

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up

docker-down:
	docker-compose -f docker/docker-compose.yml down

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
