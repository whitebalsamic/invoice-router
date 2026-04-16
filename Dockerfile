FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src

RUN python -m pip install --upgrade pip \
    && python -m pip install .

USER appuser
WORKDIR /workspace
ENTRYPOINT ["invoice-router"]
CMD ["--help"]
