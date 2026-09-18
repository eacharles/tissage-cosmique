<p align="center">
  <img src="Tissage-Cosmique-logo.jpg" alt="tissage-cosmique logo" width="400">
</p>

# tissage-cosmique

Cosmological computations with provenance tracking, emulator support, and hot-swap dispatch.

Built on [macon](https://github.com/eacharles/macon) (CRUD database framework) and [tisserande](https://github.com/eacharles/tisserande) (execution provenance tracking).

## Overview

**tissage-cosmique** wraps [pyccl](https://github.com/LSSTDESC/CCL) (the Core Cosmology Library) to:

1. **Run cosmological computations** (distances, power spectra) with automatic provenance tracking
2. **Store inputs, outputs, and provenance** in a database via tisserande's `@track` decorator
3. **Build emulators** (fast surrogate models) trained on stored computation results
4. **Hot-swap emulators** for the original code — transparently replace pyccl calls at runtime

## Installation

```bash
pip install tissage-cosmique                  # core
pip install "tissage-cosmique[pyccl]"         # pyccl computations
pip install "tissage-cosmique[emulators]"     # GP emulator (scikit-learn)
pip install "tissage-cosmique[tensorflow]"    # TensorFlow backend
pip install "tissage-cosmique[pytorch]"       # PyTorch backend
pip install "tissage-cosmique[symbolic]"      # PySR symbolic regression
pip install "tissage-cosmique[dashboard]"     # Plotly/Dash web dashboard
pip install "tissage-cosmique[dev]"           # development (tests, lint, typing)
```

## Quick Start

```python
import numpy as np
from tissage_cosmique.computations.distances import comoving_angular_distance
from tissage_cosmique.emulators import GPEmulator, build_training_data, params_to_feature_matrix
from tissage_cosmique.swap import swappable, register

# Define cosmological parameters
params = dict(Omega_c=0.2589, Omega_b=0.0486, h=0.6774,
              n_s=0.9667, sigma8=0.8159, Omega_k=0.0, w0=-1.0, wa=0.0)
a_grid = np.linspace(0.2, 0.8, 50)

# Run the computation (automatically tracked by @track)
distances = comoving_angular_distance(params, a_grid)

# Train a GP emulator
samples = [{**params, "Omega_c": np.random.uniform(0.22, 0.32),
            "h": np.random.uniform(0.62, 0.75)} for _ in range(50)]
X, y = build_training_data(comoving_angular_distance, samples, a_grid,
                           param_names=["Omega_c", "h", "sigma8"])
emu = GPEmulator(feature_names=["Omega_c", "h", "sigma8", "a"])
emu.fit(X, y)

# Hot-swap: replace pyccl with the emulator (40x faster)
compute = swappable(comoving_angular_distance)
register(comoving_angular_distance, emu, ["Omega_c", "h", "sigma8"])
result = compute(params, a_grid)  # uses the emulator
```

## Architecture

```
tissage_cosmique/
├── computations/      pyccl-wrapping functions with @track provenance
├── emulators/         training, 4 backends, validation, inversion, latent space
│   ├── backends/      GP, TensorFlow, PyTorch, PySR
│   └── codecs/        PCA and autoencoder for latent space encoding
├── swap/              hot-swap registry and dispatch
├── dashboard/         Plotly/Dash web interface
├── db/                domain tables (CosmologyParams, EmulatorRecord, CodecRecord)
├── models/            Pydantic models
├── db_oper/           macon TableOperations
├── local_async/       async session-managed operations
├── local_sync/        synchronous wrappers
├── router/            FastAPI endpoints
├── cli/               Click CLI entry points
└── config.py          Pydantic Settings configuration
```

### Three-Package Design

| Package | Role |
|---------|------|
| **macon** | Generic CRUD database framework (SQLAlchemy, FastAPI, Click) |
| **tisserande** | Execution provenance tracking (`@track` decorator, DAG storage) |
| **tissage-cosmique** | Cosmology computations, emulators, hot-swap (this package) |

## Emulator Backends

| Backend | Class | Key Feature |
|---------|-------|-------------|
| Gaussian Process | `GPEmulator` | Uncertainty estimates, best at small N |
| TensorFlow | `TensorFlowEmulator` | Scales to large N |
| PyTorch | `PyTorchEmulator` | Gradient-based inversion |
| Symbolic (PySR) | `SymbolicEmulator` | Discovers closed-form expressions |

## Latent Space Emulation

Compress output curves into a low-dimensional latent code, emulate in latent space, then decode:

```python
from tissage_cosmique.emulators import PCACodec, LatentEmulator, GPEmulator

le = LatentEmulator(
    codec=PCACodec(n_components=3),
    emulator_factory=lambda: GPEmulator(feature_names=["Omega_c", "h", "sigma8"]),
)
le.fit(samples, comoving_angular_distance, a_grid, param_names=["Omega_c", "h", "sigma8"])
curve = le.predict(params)  # full output curve from 3 latent dimensions
```

## Emulator Inversion

Recover cosmological parameters from target observations:

```python
result = emu.invert(
    y_target=observed_distances,
    free_params=["Omega_c", "h"],
    fixed_params={"sigma8": 0.8159, "a": a_grid},
    x0={"Omega_c": 0.27, "h": 0.68},
    bounds={"Omega_c": (0.2, 0.35), "h": (0.6, 0.8)},
)
print(result.x_solution)  # {"Omega_c": 0.259, "h": 0.677}
```

## Dashboard

Interactive web interface for exploring stored data:

```bash
tissage-cosmique-dashboard --db-url sqlite+aiosqlite:///my_database.db
```

4 tabs: Cosmology Parameters, Provenance Explorer, Emulator Registry, Codec Registry.

## CLI

```bash
tissage-cosmique-local          # local DB admin (CRUD for all tables)
tissage-cosmique-server         # FastAPI/uvicorn server
tissage-cosmique-dashboard      # Plotly/Dash web dashboard
```

## Examples

See the [examples/](examples/) directory for Jupyter notebooks:

- `emulator_distances.ipynb` — GP emulator for comoving angular distance
- `emulator_power_spectrum.ipynb` — multi-grid emulation of P(k, a)
- `compare_emulator_backends.ipynb` — GP vs TF vs PyTorch vs PySR comparison
- `full_pipeline.ipynb` — complete loop: DB → pyccl → provenance → emulator
- `hot_swap_demo.ipynb` — transparent emulator dispatch (40x speedup)
- `inversion_demo.ipynb` — parameter recovery from observations
- `latent_emulator_demo.ipynb` — PCA and autoencoder latent space emulation

## Development

```bash
pip install -e ".[dev]"
pytest tests/                    # run test suite
ruff check src/ tests/           # lint
mypy src/                        # type check
cd docs && make html             # build documentation
```

## License

MIT — see [LICENSE](LICENSE).
