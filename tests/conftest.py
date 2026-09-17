import asyncio

import pytest

from tissage_cosmique.db.base import Base, close_db, init_db

DB_URL = "sqlite+aiosqlite://"


@pytest.fixture(autouse=True)
def _setup_db():
    """Initialize DB for each test with in-memory SQLite."""
    asyncio.run(close_db())
    init_db(DB_URL)

    async def _create_tables():
        from macon.db.session import get_session

        async with get_session() as session:
            conn = await session.connection()
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_create_tables())
    yield
    asyncio.run(close_db())
