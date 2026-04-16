# Testing

The repository uses three test layers:

- `tests/unit`: isolated logic and CLI wiring
- `tests/integration`: real adapter and multi-module coverage
- `tests/e2e`: top-level public flows with minimal internal mocking

Useful commands:

```bash
make unit
make integration
make e2e
make test
```

Equivalent raw pytest commands:

```bash
pytest -m "not integration and not e2e"
pytest -m integration
pytest -m e2e
pytest
```

CI also runs:

- `ruff check .`
- `ruff format --check .`
- `mypy`
- `python -m build`
- `docker build .`
