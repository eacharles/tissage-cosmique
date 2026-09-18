# CLAUDE.md — tissage-cosmique

## Overview

`tissage-cosmique` is a Python package for cosmological computations with automatic provenance tracking, emulator support, and hot-swap dispatch. It wraps [pyccl](https://github.com/LSSTDESC/CCL) (the Core Cosmology Library) to:

1. **Run cosmological computations** (distances, power spectra) via pyccl with `@track` provenance
2. **Store inputs, outputs, and provenance** in a database via tisserande + macon
3. **Build emulators** (GP, TensorFlow, PyTorch, PySR) trained on stored results
4. **Hot-swap emulators for original code** — transparently replace pyccl calls at runtime
5. **Invert emulators** — recover input parameters from target outputs
6. **Latent space emulation** — compress output curves via PCA or autoencoder codecs
7. **Track emulator/codec artifacts** in the database alongside provenance
8. **Visualize** via an interactive Plotly/Dash dashboard

The package builds on two sibling libraries:
- **macon** (`/Users/echarles/software/KIPAC/macon`) — layered CRUD database framework (SQLAlchemy ORM → operations → async/sync wrappers → FastAPI routers → HTTP clients)
- **tisserande** (`/Users/echarles/software/KIPAC/tisserande`) — execution provenance tracking (decorators, DAG storage, argument inspection)

All cosmology/astrophysics-specific code lives here. Generic framework features belong in macon or tisserande.

## Architecture

```
tissage_cosmique/
├── computations/         pyccl-wrapping functions decorated with @track
│   ├── cosmology.py      make_pyccl_cosmology helper (dict → pyccl.Cosmology)
│   ├── distances.py      comoving_angular_distance(cosmo_params, a)
│   └── power.py          linear_matter_power(cosmo_params, k, a)
├── emulators/            training, backends, validation, inversion, latent space
│   ├── base.py           Emulator ABC (fit/predict/save/load/invert/metadata)
│   ├── training.py       build_training_data, params_to_feature_matrix (multi-grid)
│   ├── validation.py     validate_emulator, cross_validate, check_calibration
│   ├── inversion.py      InversionResult, invert_minimize (scipy)
│   ├── db_training.py    query_computation_results, run_tracked_batch
│   ├── latent.py         LatentEmulator (codec + K per-dimension emulators)
│   ├── backends/         GP, TensorFlow, PyTorch, PySR — all with invert() overrides
│   │   ├── gp.py         GPEmulator (scikit-learn, uncertainty-weighted inversion)
│   │   ├── tensorflow.py TensorFlowEmulator (Keras, GradientTape inversion)
│   │   ├── pytorch.py    PyTorchEmulator (autograd inversion, seed support)
│   │   └── symbolic.py   SymbolicEmulator (PySR, sympy.solve inversion)
│   └── codecs/           latent space encoder/decoder
│       ├── base.py       Codec ABC (fit/encode/decode/save/load)
│       ├── pca.py        PCACodec (sklearn PCA, explained_variance)
│       └── autoencoder.py AutoencoderCodec (PyTorch encoder/decoder)
├── swap/                 hot-swap machinery
│   ├── registry.py       register/unregister/enable/disable/reset
│   └── dispatch.py       swappable() wrapper, original() context manager
├── dashboard/            Plotly/Dash web interface (4 tabs)
│   ├── app.py            Dash app factory with tabs
│   ├── callbacks.py      execution detail + DAG visualization
│   ├── queries.py        DB → pandas DataFrames
│   └── components.py     reusable Plotly/Dash builders
├── db/                   domain ORM tables
│   ├── base.py           re-exports macon Base, init_db, get_session, close_db
│   ├── cosmology.py      CosmologyParamsTable
│   ├── codec_record.py   CodecRecordTable
│   ├── emulator_record.py EmulatorRecordTable (FK to codec_record)
│   └── utils.py          uuid7() helper
├── models/               Pydantic models (Base → Create → Response pattern)
├── db_oper/              TableOperations singletons
├── local_async/          async session-managed operations
├── local_sync/           synchronous wrappers (asyncio.run)
├── router/               FastAPI app with CRUD endpoints
├── cli/                  Click CLI entry points
│   ├── local/            tissage-cosmique-local (DB admin)
│   ├── server/           tissage-cosmique-server (FastAPI/uvicorn)
│   └── dashboard/        tissage-cosmique-dashboard (Plotly/Dash)
└── config.py             Pydantic Settings (TISSAGE_COSMIQUE__ prefix)
```

### Data Flow

```
CosmologyParams (macon CRUD)
  → comoving_angular_distance / linear_matter_power (pyccl, @track provenance)
    → execution/node/edge records (tisserande DB)
      → query_computation_results (reconstruct training data from DB)
        → Emulator.fit / LatentEmulator.fit (train surrogate)
          → EmulatorRecord / CodecRecord (register in DB)
            → swappable() dispatch (hot-swap at runtime)
              → Emulator.invert (recover parameters from observations)
```

### Relationship to Sibling Packages

| Concern | Where it lives |
|---------|---------------|
| CRUD operations, DB sessions, filtering, routers, CLI scaffolding | macon |
| Provenance tracking (@track), execution DAG, argument inspection | tisserande |
| Cosmology functions, emulators, codecs, hot-swap, inversion, dashboard | tissage-cosmique |

When new generic capabilities are needed (e.g., a new filter operator, an async tracking backend), extend macon or tisserande. Domain logic stays here.

## Key Design Decisions

- **Tracked computations**: Every pyccl-wrapping function uses `@track` from tisserande. Inputs annotated with `ConfigDict[dict]`, `ArrayArg[np.ndarray]`, `Param[float]`.
- **Emulator ABC**: `fit(X, y)`, `predict(X)`, `save/load`, `invert()`, `metadata`. Four backends implement this.
- **Inversion**: Default via `scipy.optimize.minimize`. GP overrides with uncertainty-weighted objective. PyTorch/TF override with autograd. PySR overrides with `sympy.solve`.
- **Multi-grid training**: `build_training_data` accepts `*grids` and `result_axes_order` for N-D outputs (e.g., P(k, a) → `(n_a, n_k)`).
- **Latent space**: `Codec` ABC (PCA, autoencoder) + `LatentEmulator` wrapping codec + K scalar emulators.
- **Hot-swap**: `swappable(fn)` wrapper checks global registry; `original(fn)` context manager bypasses.
- **DB tracking**: `EmulatorRecord` and `CodecRecord` tables with FK to each other and `execution_ids` linking to tisserande provenance.
- **NN determinism**: PyTorch and TensorFlow backends accept a `seed` parameter for reproducible training.
- **Three-model pattern**: `FooBase` → `FooCreate` → `Foo` (response with `id_`, `ConfigDict(from_attributes=True)`).
- **UUID7 primary keys** for domain tables (consistent with tisserande).
- **Macon layer stack**: New tables follow `models/` → `db/` → `db_oper/` → `local_async/` → `local_sync/` → `router/` → `cli/`.
- **Defensive guards**: Use `macon.common.unexpected()` for conditions that should never be true, `except ... as uexc:` for exceptions that should never fire. Both excluded from coverage.

## Development

### Setup
```bash
pip install -e ".[dev]"
```

### Dependencies
- Core: `tisserande`, `macon[db]`, `numpy`
- Computations: `pyccl` (+ `camb` for power spectra)
- Emulators: `scikit-learn`, `joblib`
- Optional backends: `tensorflow`, `torch`, `pysr`
- Dashboard: `dash`, `plotly`, `pandas`
- Server: `macon[server]` (FastAPI)
- Dev: `pytest`, `pytest-asyncio`, `pytest-cov`, `mypy`, `ruff`, `pylint`

### Testing
```bash
pytest tests/                    # full test suite (~135 tests, 99% coverage)
pytest tests/ -x --tb=short     # stop at first failure
pytest tests/ --cov=src/tissage_cosmique --cov-report=term-missing
```

Tests use in-memory SQLite (`sqlite+aiosqlite://`). The `conftest.py` fixture imports `tisserande.db` to register provenance tables, calls `init_db()` + `Base.metadata.create_all`, and resets tracking between tests.

### Linting & Type Checking
```bash
ruff check src/ tests/
ruff format src/ tests/
mypy src/                        # 62 source files
```

### Documentation
```bash
pip install -e ".[docs]"
cd docs && make html             # builds to docs/_build/html/
```
Sphinx with `sphinx-autoapi`, `sphinx_rtd_theme`, NumPy-style docstrings. ReadTheDocs config in `.readthedocs.yaml`.

### Configuration
- Line length: 110 (ruff), 120 (pylint)
- Python: 3.13+ required
- Build: setuptools + setuptools_scm
- Ruff ignores: `COM812`, `N802-N816`, `UP047` (matching macon/tisserande)
- Env prefix: `TISSAGE_COSMIQUE__` (e.g., `TISSAGE_COSMIQUE__DB__URL`)
- Default DB: `sqlite+aiosqlite:///tissage_cosmique.db`
- Pytest: `filterwarnings` ignores `ConvergenceWarning`, `DeprecationWarning`, `FutureWarning`, `UserWarning`
- Coverage: dashboard module excluded via `omit` (requires running server)

### Coverage Exclusion Patterns
```
"pragma: no cover"
"unexpected"
"as uexc:"
"def __repr__"
"def __str__"
"raise AssertionError"
"raise NotImplementedError"
```

### CLI Entry Points
- `tissage-cosmique-local` — local DB admin (CRUD for CosmologyParams, EmulatorRecord, CodecRecord)
- `tissage-cosmique-server` — FastAPI/uvicorn server
- `tissage-cosmique-dashboard` — Plotly/Dash web interface

### CI/CD
GitHub Actions workflows in `.github/workflows/`:
- `linting.yml` — ruff + mypy on push/PR to main
- `testing-and-coverage.yml` — pytest on push/PR + weekly schedule
- `docs.yml` — Sphinx build on docs/src changes
- `publish-to-pypi.yml` — trusted publishing (OIDC) on GitHub release

### Adding a New Computation

1. Create a function in `computations/` that wraps the pyccl call
2. Decorate with `@track` and annotate arguments (`ConfigDict[dict]`, `ArrayArg[np.ndarray]`, `Param[float]`)
3. Lazy-import pyccl inside the function so the module is importable without pyccl
4. Use `make_pyccl_cosmology(params)` to convert the dict
5. Add tests in `tests/` with `pytest.importorskip("pyccl")`

### Adding a New Emulator Backend

1. Subclass `Emulator` in `emulators/backends/`
2. Implement `fit`, `predict`, `save`, `load`, `metadata`
3. Optionally override `invert()` with a backend-specific method
4. Add `seed` parameter for deterministic training if applicable
5. Export from `emulators/backends/__init__.py` and `emulators/__init__.py`
6. Add tests validating accuracy, save/load roundtrip, and inversion

### Adding a New Domain Table

1. Create Pydantic models (Base/Create/Response) in `models/`
2. Create ORM table in `db/` inheriting from `Base`, with 3 classmethods
3. Create `TableOperations` singleton in `db_oper/`
4. Create `LocalOperations` singleton in `local_async/`
5. Create `SyncOperations` singleton in `local_sync/`
6. Update all `__init__.py` exports
7. Add router in `router/app.py` and CLI command in `cli/local/top.py`

### Example Notebooks

7 notebooks in `examples/`:
- `emulator_distances.ipynb` — GP emulator for comoving angular distance
- `emulator_power_spectrum.ipynb` — multi-grid emulation with ARD kernel
- `compare_emulator_backends.ipynb` — GP vs TF vs PyTorch vs PySR
- `full_pipeline.ipynb` — complete DB → pyccl → provenance → emulator loop
- `hot_swap_demo.ipynb` — transparent dispatch (40x speedup)
- `inversion_demo.ipynb` — parameter recovery with loss landscape
- `latent_emulator_demo.ipynb` — PCA and autoencoder latent space
