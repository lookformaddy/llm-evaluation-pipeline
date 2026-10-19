# LLM Evaluation Pipeline — slim, offline-capable image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for layer caching.
COPY pyproject.toml requirements.txt README.md LICENSE ./
COPY src ./src
RUN pip install -e ".[api,dev]"

# Project payload: bundled demo knowledge base, golden dataset, examples.
COPY data ./data
COPY examples ./examples
COPY tests ./tests
COPY docs ./docs

# Default: run the test suite. Override for the demo, e.g.:
#   docker run --rm llm-eval python -m llm_eval_pipeline.cli demo --out /tmp/demo
CMD ["pytest", "-q"]
