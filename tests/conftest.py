import asyncio
import os
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

import asyncpg
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from invoice_router.infrastructure.persistence.postgres import (
    benchmark_postgres_dsn,
    recreate_postgres_database,
)


def pytest_collection_modifyitems(config, items):
    for item in items:
        path = Path(str(item.fspath))
        parts = set(path.parts)
        if "tests" not in parts:
            continue
        if "unit" in parts:
            item.add_marker(pytest.mark.unit)
        elif "integration" in parts:
            item.add_marker(pytest.mark.integration)
        elif "e2e" in parts:
            item.add_marker(pytest.mark.e2e)

        if "postgres_database_url" in item.fixturenames:
            item.add_marker(pytest.mark.postgres)


def _drop_postgres_database(target_dsn: str) -> None:
    parsed = urlparse(target_dsn)
    database_name = parsed.path.lstrip("/")
    admin_dsn = urlunparse(parsed._replace(path="/postgres"))

    async def _drop() -> None:
        conn = await asyncpg.connect(admin_dsn)
        try:
            await conn.execute(f'DROP DATABASE IF EXISTS "{database_name}" WITH (FORCE)')
        finally:
            await conn.close()

    asyncio.run(_drop())


@pytest.fixture
def postgres_database_url():
    candidates = []
    if env_dsn := os.environ.get("DATABASE_URL"):
        candidates.append(env_dsn)
    candidates.extend(
        [
            "postgresql://invoice_router:invoice_router@localhost:5432/invoice_router",
            "postgresql://ir3:ir3@localhost:5432/ir3",
        ]
    )

    last_error = None
    for base_dsn in candidates:
        target_dsn = benchmark_postgres_dsn(base_dsn, f"pytest_{uuid4().hex}")
        try:
            recreate_postgres_database(target_dsn)
        except (
            Exception
        ) as exc:  # pragma: no cover - exercised only when a local DSN is unavailable
            last_error = exc
            continue
        try:
            yield target_dsn
        finally:
            _drop_postgres_database(target_dsn)
        return

    if last_error is not None:
        raise last_error
    raise RuntimeError("No usable Postgres test DSN was available.")
