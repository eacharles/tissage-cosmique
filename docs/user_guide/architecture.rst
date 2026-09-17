Architecture
============

Three-Package Design
--------------------

tissage-cosmique builds on two sibling libraries:

- **macon** — generic CRUD database framework (SQLAlchemy ORM, operations layers,
  FastAPI routers, Click CLI scaffolding)
- **tisserande** — execution provenance tracking (``@track`` decorator, DAG storage,
  argument inspection)

All cosmology and astrophysics-specific code lives in tissage-cosmique. Generic
framework features belong in macon or tisserande.

Package Layout
--------------

.. code-block:: text

    tissage_cosmique/
    ├── computations/      pyccl-wrapping functions decorated with @track
    │   ├── cosmology.py   Cosmology object creation helper
    │   ├── distances.py   comoving_angular_distance
    │   └── power.py       linear_matter_power
    ├── emulators/         emulator training, backends, and validation
    │   ├── base.py        Emulator ABC
    │   ├── training.py    build_training_data, params_to_feature_matrix
    │   ├── validation.py  validate_emulator, cross_validate, check_calibration
    │   ├── db_training.py query_computation_results, run_tracked_batch
    │   └── backends/      GP, TensorFlow, PyTorch, PySR
    ├── swap/              hot-swap machinery
    │   ├── registry.py    register/unregister/enable/disable emulators
    │   └── dispatch.py    swappable() wrapper, original() context manager
    ├── db/                domain-specific ORM tables (CosmologyParams)
    ├── models/            Pydantic models
    ├── db_oper/           TableOperations (macon layer)
    ├── local_async/       async session-managed operations
    ├── local_sync/        synchronous wrappers
    ├── router/            FastAPI endpoints
    ├── cli/               Click CLI entry points
    └── config.py          Pydantic Settings configuration

Data Flow
---------

.. code-block:: text

    User calls tracked computation (e.g., compute_power_spectrum)
      → tisserande @track captures inputs/outputs as provenance DAG
      → results stored in DB (macon CRUD layer)
      → emulator trainer queries DB for (parameter, result) pairs
      → trains surrogate model, stores emulator artifact
      → hot-swap registry maps function → emulator
      → subsequent calls routed through emulator (if enabled)

Layer Stack
-----------

Every database table follows the macon layer stack:

.. code-block:: text

    models/        Pydantic models (Base → Create → Response)
    db/            SQLAlchemy ORM table
    db_oper/       TableOperations (CRUD + validation)
    local_async/   Session-managed async operations
    local_sync/    Synchronous wrappers (asyncio.run)

All layers use the same three generic type parameters:
``[OrmTable, ResponseModel, CreateModel]``.
