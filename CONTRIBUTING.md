# Contributing

Thanks for taking a look at `invoice-router`.

## Local setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
```

Install Tesseract locally for native development:

```bash
brew install tesseract
```

If you prefer containers, use the documented `docker compose` workflow instead.

## Common commands

```bash
make lint
make typecheck
make test
make unit
make integration
make e2e
make demo
make build
```

## Project expectations

- Keep public-facing docs and examples aligned with the supported CLI.
- Prefer deterministic improvements over opaque heuristics when routing or validation behavior changes.
- Add or update tests with behavior changes, especially when modifying routing, normalization, or persistence behavior.
- Avoid committing real invoice data or secrets.

## Pull requests

- Keep PRs focused.
- Include a short summary of the behavior change and the verification you ran.
- Update docs if a new command, config key, or workflow becomes part of the public surface.

## GitHub labels

Suggested label categories for triage:

- `bug`
- `enhancement`
- `documentation`
- `good first issue`
- `help wanted`
- `infra`
- `testing`
