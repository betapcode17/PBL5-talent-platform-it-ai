# app/services/pg_db.py
"""PostgreSQL connection helper using asyncpg.
Provides a pool and a small helper to run the DDL migration file.
"""
import asyncio
import logging
from app.config import DATABASE_URL
from pathlib import Path

# Try to import asyncpg; if unavailable, fall back to psycopg2 synchronous calls
try:
    import asyncpg  # type: ignore
except Exception:
    asyncpg = None  # type: ignore

try:
    import psycopg2
    import psycopg2.extras
except Exception:
    psycopg2 = None  # type: ignore

_pool = None
_use_sync = False

async def init_pg_pool(min_size: int = 1, max_size: int = 10):
    """Initialize an asyncpg pool if available, otherwise use sync psycopg2 fallback.

    Returns a pool-like object for async use.
    """
    global _pool, _use_sync
    if _pool is not None:
        return _pool

    if asyncpg is not None:
        logging.info("Initializing asyncpg pool")
        _pool = await asyncpg.create_pool(dsn=DATABASE_URL, min_size=min_size, max_size=max_size)
        _use_sync = False
        return _pool

    if psycopg2 is not None:
        logging.warning("asyncpg not available; using psycopg2 sync fallback via threads")
        _use_sync = True
        _pool = DATABASE_URL  # store DSN for sync fallback
        return _pool

    raise RuntimeError("No PostgreSQL driver available (asyncpg or psycopg2 required)")

async def close_pg_pool():
    global _pool, _use_sync
    if _pool is None:
        return
    if not _use_sync and asyncpg is not None:
        await _pool.close()  # type: ignore
    _pool = None
    _use_sync = False

async def run_migration_sql():
    """Run the migrations/pg_chat_schema.sql file to create tables if missing."""
    base = Path(__file__).resolve().parent.parent.parent
    sql_file = base / "migrations" / "pg_chat_schema.sql"
    if not sql_file.exists():
        logging.warning("Migration SQL file not found: %s", sql_file)
        return
    sql = sql_file.read_text()
    pool = await init_pg_pool()
    if not _use_sync and asyncpg is not None:
        async with pool.acquire() as conn: # type: ignore
            await conn.execute(sql)
    else:
        # run in thread to avoid blocking event loop
        def _run_sync():
            conn = psycopg2.connect(DATABASE_URL) # type: ignore
            try:
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(sql)
            finally:
                conn.close()

        await asyncio.to_thread(_run_sync)


def ensure_pg_running():
    """Synchronous helper to init pool and run migrations. Useful for quick scripts."""
    asyncio.get_event_loop().run_until_complete(_ensure_pg())


async def _ensure_pg():
    await init_pg_pool()
    await run_migration_sql()


async def fetch(query: str, *args):
    pool = await init_pg_pool()
    if not _use_sync and asyncpg is not None:
        async with pool.acquire() as conn: # type: ignore
            return await conn.fetch(query, *args)

    def _run():
        conn = psycopg2.connect(DATABASE_URL) # type: ignore
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur: # type: ignore
                cur.execute(query, args)
                return cur.fetchall()
        finally:
            conn.close()

    return await asyncio.to_thread(_run)


async def fetchrow(query: str, *args):
    rows = await fetch(query, *args)
    return rows[0] if rows else None


async def execute(query: str, *args):
    pool = await init_pg_pool()
    if not _use_sync and asyncpg is not None:
        async with pool.acquire() as conn: # type: ignore
            return await conn.execute(query, *args)

    def _run():
        conn = psycopg2.connect(DATABASE_URL) # type: ignore
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(query, args)
                return cur.statusmessage
        finally:
            conn.close()

    return await asyncio.to_thread(_run)
