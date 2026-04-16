import asyncio
import re
from typing import Any, Iterable, Optional
from urllib.parse import urlparse, urlunparse

import asyncpg

POSTGRES_PREFIXES = ("postgresql://", "postgres://")


def is_postgres_target(target: Optional[str]) -> bool:
    return bool(target) and str(target).startswith(POSTGRES_PREFIXES)


def translate_query(sql: str) -> str:
    parts = sql.split("?")
    if len(parts) == 1:
        return sql
    translated = [parts[0]]
    for index, part in enumerate(parts[1:], start=1):
        translated.append(f"${index}")
        translated.append(part)
    return "".join(translated)


class PostgresClient:
    def __init__(self, dsn: str):
        self.dsn = dsn

    def execute(self, sql: str, params: Iterable[Any] = ()) -> str:
        return asyncio.run(self._execute(sql, tuple(params)))

    def fetchone(self, sql: str, params: Iterable[Any] = ()) -> Optional[dict[str, Any]]:
        return asyncio.run(self._fetchone(sql, tuple(params)))

    def fetchall(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        return asyncio.run(self._fetchall(sql, tuple(params)))

    async def _connect(self) -> asyncpg.Connection:
        return await asyncpg.connect(self.dsn)

    async def _execute(self, sql: str, params: tuple[Any, ...]) -> str:
        conn = await self._connect()
        try:
            return await conn.execute(translate_query(sql), *params)
        finally:
            await conn.close()

    async def _fetchone(self, sql: str, params: tuple[Any, ...]) -> Optional[dict[str, Any]]:
        conn = await self._connect()
        try:
            row = await conn.fetchrow(translate_query(sql), *params)
            return dict(row) if row is not None else None
        finally:
            await conn.close()

    async def _fetchall(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        conn = await self._connect()
        try:
            rows = await conn.fetch(translate_query(sql), *params)
            return [dict(row) for row in rows]
        finally:
            await conn.close()


def benchmark_postgres_dsn(base_dsn: str, suffix: str) -> str:
    parsed = urlparse(base_dsn)
    base_name = parsed.path.lstrip("/") or "invoice_router"
    sanitized_suffix = re.sub(r"[^a-z0-9_]+", "_", suffix.lower()).strip("_") or "benchmark"
    db_name = f"{base_name}_{sanitized_suffix}"
    return urlunparse(parsed._replace(path=f"/{db_name}"))


def recreate_postgres_database(target_dsn: str) -> None:
    parsed = urlparse(target_dsn)
    database_name = parsed.path.lstrip("/")
    admin_dsn = urlunparse(parsed._replace(path="/postgres"))
    asyncio.run(_recreate_postgres_database(admin_dsn, database_name))


async def _recreate_postgres_database(admin_dsn: str, database_name: str) -> None:
    conn = await asyncpg.connect(admin_dsn)
    try:
        await conn.execute(f'DROP DATABASE IF EXISTS "{database_name}" WITH (FORCE)')
        await conn.execute(f'CREATE DATABASE "{database_name}"')
    finally:
        await conn.close()
