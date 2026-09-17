Configuration
=============

tissage-cosmique uses Pydantic Settings with environment variable overrides.

Environment Variables
---------------------

All settings use the prefix ``TISSAGE_COSMIQUE__`` with ``__`` as the nesting
delimiter:

.. list-table::
   :header-rows: 1

   * - Variable
     - Default
     - Description
   * - ``TISSAGE_COSMIQUE__DB__URL``
     - ``sqlite+aiosqlite:///tissage_cosmique.db``
     - Database URL
   * - ``TISSAGE_COSMIQUE__DB__ECHO``
     - ``false``
     - SQLAlchemy engine echo
   * - ``TISSAGE_COSMIQUE__EMULATOR__ENABLED``
     - ``false``
     - Enable emulator hot-swap globally

Database Setup
--------------

The database is initialized lazily. For scripts::

    from tissage_cosmique.db.base import init_db, get_session, close_db, Base
    import asyncio

    init_db("sqlite+aiosqlite:///my_database.db")

    async def setup():
        async with get_session() as session:
            conn = await session.connection()
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(setup())

Since macon, tisserande, and tissage-cosmique share the same ``Base``, a single
``create_all`` creates all tables.

CLI Entry Points
----------------

::

    tissage-cosmique-local   # local DB admin (CRUD for CosmologyParams)
    tissage-cosmique-server  # FastAPI/uvicorn server
