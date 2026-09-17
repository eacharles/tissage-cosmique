# CLAUDE.md — tissage-cosmique

## Overview

`tissage-cosmique` is a Python package for cosmological computations with automatic provenance tracking and emulator support. It wraps [pyccl](https://github.com/LSSTDESC/CCL) (the Core Cosmology Library) to:

1. **Run cosmological computations** (power spectra, correlation functions, distances, etc.) via pyccl
2. **Store inputs, outputs, and provenance** in a database via tisserande's tracking decorators
3. **Build emulators** (fast surrogate models) trained on the stored computation results
4. **Hot-swap emulators for original code** — transparently replace pyccl calls with emulator predictions at runtime

The package builds on two sibling libraries:
- **macon** (`/Users/echarles/software/KIPAC/macon`) — layered CRUD database framework (SQLAlchemy ORM → operations → async/sync wrappers → FastAPI routers → HTTP clients)
- **tisserande** (`/Users/echarles/software/KIPAC/tisserande`) — execution provenance tracking (decorators, DAG storage, argument inspection)

All cosmology/astrophysics-specific code lives here. Generic framework features belong in macon or tisserande.

## Architecture

```
tissage-cosmique
├── computations/      ← pyccl-wrapping functions decorated with @track
│   ├── cosmology.py   ← Cosmology object creation
│   ├── power.py       ← power spectra (linear, nonlinear)
│   ├── distances.py   ← distance measures (comoving, luminosity, angular diameter)
│   ├── growth.py      ← growth factor, growth rate
│   └── correlations.py ← two-point correlation functions
├── db/                ← domain-specific ORM tables (cosmology params, computation results)
├── models/            ← Pydantic models for cosmology parameters and results
├── db_oper/           ← TableOperations subclasses (inheriting from macon)
├── local_async/       ← session-managed async operations
├── local_sync/        ← synchronous wrappers
├── emulators/         ← emulator training, storage, and inference
│   ├── base.py        ← Emulator protocol / base class
│   ├── registry.py    ← emulator registry and hot-swap dispatch
│   ├── training.py    ← train emulators from stored computation results
│   └── backends/      ← concrete emulator implementations (GP, neural net, polynomial, etc.)
├── swap/              ← hot-swap machinery
│   ├── registry.py    ← maps computation functions → emulators
│   └── dispatch.py    ← transparent call routing (original vs emulator)
├── config.py          ← package configuration (Pydantic Settings)
├── router/            ← FastAPI endpoints for running computations and querying results
└── cli/               ← CLI entry points
```

### Data Flow

```
User calls tracked computation (e.g., compute_power_spectrum)
  → tisserande @track captures inputs/outputs as provenance DAG
  → results stored in DB (macon CRUD layer)
  → emulator trainer queries DB for (parameter, result) pairs
  → trains surrogate model, stores emulator artifact
  → hot-swap registry maps function → emulator
  → subsequent calls routed through emulator (if enabled)
```

### Relationship to Sibling Packages

| Concern | Where it lives |
|---------|---------------|
| CRUD operations, DB sessions, filtering, routers, CLI scaffolding | macon |
| Provenance tracking (@track), execution DAG, argument inspection | tisserande |
| Cosmology functions, pyccl wrappers, emulator training/inference, hot-swap logic | tissage-cosmique |

When new generic capabilities are needed (e.g., a new filter operator, an async tracking backend), extend macon or tisserande. Domain logic (cosmological parameters, pyccl calls, emulator architectures) stays here.

## Key Design Decisions

- **Tracked computations**: Every pyccl-wrapping function uses `@track` from tisserande so provenance is captured automatically. Inputs are annotated with `Param[T]`, `ConfigDict[T]`, `ArrayArg[T]`, etc.
- **Emulator protocol**: Emulators implement a common interface (`predict`, `save`, `load`, `metadata`) so backends are interchangeable.
- **Hot-swap via registry**: A global registry maps `(module, function_name)` → emulator. When enabled, the swap dispatcher intercepts calls and routes to the emulator. When disabled or when the emulator has no coverage for the input domain, the original pyccl function is called.
- **Three-model pattern**: Following macon/tisserande convention — `FooBase` (shared fields) → `FooCreate` → `Foo` (response with `id_`, `ConfigDict(from_attributes=True)`).
- **UUID7 primary keys** for domain tables (consistent with tisserande).
- **Layered operations**: New tables follow the macon layer stack: `db/` → `db_oper/` → `local_async/` → `local_sync/`.

## Development

### Setup
```bash
pip install -e ".[dev]"
```

### Dependencies
- Core: `pyccl`, `tisserande`, `macon[db]`
- Emulators: `scikit-learn`, `numpy` (and optionally `torch` for neural net backends)
- Server: `macon[server]` (FastAPI)
- Dev: `pytest`, `pytest-asyncio`, `pytest-cov`, `mypy`, `ruff`, `pylint`

### Testing
```bash
pytest tests/                    # full test suite
pytest tests/ -x --tb=short     # stop at first failure
pytest tests/ --cov=tissage_cosmique --cov-report=term-missing
```

Tests use in-memory SQLite (`sqlite+aiosqlite://`). Fixtures call `init_db()` and create all tables.

### Linting & Type Checking
```bash
ruff check src/ tests/
ruff format src/ tests/
mypy src/
pylint src/
```

### Configuration
- Line length: 110 (ruff), 120 (pylint)
- Python: 3.13+ required
- Build: setuptools + setuptools_scm
- Ruff ignores: `COM812`, `N802-N816`, `UP047` (matching macon/tisserande conventions)
- Env prefix: `TISSAGE_COSMIQUE__` (e.g., `TISSAGE_COSMIQUE__DB__URL`)
- Default DB: `sqlite+aiosqlite:///tissage_cosmique.db`

### Coverage Exclusion Patterns
```
"pragma: no cover"
"unexpected"
"as uexc:"
"raise AssertionError"
"raise NotImplementedError"
```

### Adding a New Computation

1. Create a function in `computations/` that wraps the pyccl call
2. Decorate with `@track` and annotate arguments with tisserande types (`Param`, `ArrayArg`, `ConfigDict`, etc.)
3. If new parameter types are needed, add Pydantic models in `models/` and ORM tables in `db/`
4. Follow the macon layer stack for any new tables: `db/` → `db_oper/` → `local_async/` → `local_sync/`
5. Add tests covering both the computation itself and the provenance it generates

### Adding a New Emulator Backend

1. Implement the emulator protocol in `emulators/backends/`
2. Register the backend type in `emulators/registry.py`
3. Ensure `predict()` returns the same shape/type as the original computation
4. Add tests validating accuracy against pyccl reference outputs
