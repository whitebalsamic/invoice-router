# invoice-router

[![CI](https://github.com/whitebalsamic/invoice-router/actions/workflows/ci.yml/badge.svg)](https://github.com/whitebalsamic/invoice-router/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`invoice-router` is a deterministic invoice processing toolkit for local datasets. It fingerprints layouts, groups related documents into reusable template families, routes extraction deliberately, and records run history for repeatable analysis.

This repository is optimized for teams who want something more predictable than a generic OCR-only pipeline:

- Fingerprint- and template-family-aware routing before extraction
- Native PDF and OCR-backed extraction paths
- Normalization and validation focused on invoice structure
- Postgres-backed processing history and benchmark comparison tools
- A public demo command that generates and processes a synthetic invoice end to end

## Built With

- [`Click`](https://click.palletsprojects.com/) for the CLI surface
- [`Pydantic`](https://docs.pydantic.dev/) and `pydantic-settings` for typed config
- [`PyMuPDF`](https://pymupdf.readthedocs.io/) for PDF handling and demo generation
- [`pytesseract`](https://pypi.org/project/pytesseract/) and Tesseract OCR for extraction
- [`opencv-python-headless`](https://pypi.org/project/opencv-python-headless/) for image preprocessing
- [`Postgres`](https://www.postgresql.org/) and [`Redis`](https://redis.io/) for run history and coordination
- [`Celery`](https://docs.celeryq.dev/) for worker-oriented execution paths

## Quickstart

### Docker quickstart

```bash
docker compose up -d postgres redis
docker compose run --rm app demo --workspace /workspace/.demo
```

That command will:

- generate a synthetic invoice PDF
- run the full pipeline locally
- write JSON output under `.demo/output/`

### Native Python quickstart

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
brew install tesseract
cp .env.example .env
make demo
```

## Preparing your data

For a real dataset, place each invoice PDF or image next to an optional GT sidecar JSON that uses
the same filename stem:

```text
my-dataset/
  invoice-001.pdf
  invoice-001.json
  invoice-002.png
  invoice-002.json
  invoice-003.pdf
```

`invoice-router process ...` works with or without GT. When GT is present, the pipeline can use it
for validation, benchmarking, and discovery-oriented evaluation.

For new datasets, prefer the canonical `gt-v2` format described in the schema and onboarding docs.
A simpler flat JSON sidecar can still be normalized for basic use when it contains recognized
invoice fields, but it is less expressive and may be ignored if no supported fields are found.

See [Ground truth and data prep](docs/ground-truth.md) for folder layout, schema guidance, minimal
examples, and `invoice-router sync-ground-truth` usage.

## Public CLI

```bash
invoice-router --help
invoice-router --version
invoice-router demo --workspace .demo
invoice-router process /path/to/invoices --show-per-invoice
invoice-router families
invoice-router family-show TEMPLATE_FAMILY_ID
```

Use `--config path/to/config.yaml` or `INVOICE_ROUTER_CONFIG` to override the packaged defaults.

## Docs

- [Production workflow](PRODUCTION-WORKFLOW.md)
- [Architecture](docs/architecture.md)
- [Ground truth and data prep](docs/ground-truth.md)
- [Glossary](docs/glossary.md)
- [Environment](docs/environment.md)
- [Local services](docs/postgres.md)
- [Testing](docs/testing.md)

## Privacy and Data Handling

- Processing is local by default. This repository does not require a hosted OCR or LLM service for its core runtime.
- Inputs remain on the local filesystem. Output artifacts are written to `OUTPUT_DIR`.
- Postgres stores processing results, template-family history, and analysis runs.
- Redis is used for counters and coordination. The quickstart Compose stack runs it locally.
- The repository ships a synthetic demo flow so you can validate the toolchain without using real invoices.

## Development

```bash
make install
make lint
make typecheck
make test
make build
```

Pre-commit hooks are configured via `.pre-commit-config.yaml`.

## Limitations

- The project currently targets invoice-like datasets on local filesystems rather than generic document ingestion APIs.
- OCR quality still depends on scan quality, layout consistency, and Tesseract availability.
- The Docker setup is designed for reproducible onboarding, not a hardened production deployment.
