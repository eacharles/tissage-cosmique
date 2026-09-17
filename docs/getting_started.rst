Getting Started
===============

Installation
------------

Install the core package::

    pip install tissage-cosmique

With optional dependencies::

    pip install "tissage-cosmique[pyccl]"       # pyccl computations
    pip install "tissage-cosmique[emulators]"    # GP emulator (scikit-learn)
    pip install "tissage-cosmique[tensorflow]"   # TensorFlow backend
    pip install "tissage-cosmique[pytorch]"      # PyTorch backend
    pip install "tissage-cosmique[symbolic]"     # PySR symbolic regression

For development::

    pip install "tissage-cosmique[dev]"

Quick Example
-------------

Run a cosmological distance computation, train an emulator, and hot-swap::

    import numpy as np
    from tissage_cosmique.computations.distances import comoving_angular_distance
    from tissage_cosmique.emulators import GPEmulator, build_training_data, params_to_feature_matrix
    from tissage_cosmique.swap import swappable, register

    # 1. Define cosmological parameters and scale factors
    params = dict(Omega_c=0.2589, Omega_b=0.0486, h=0.6774,
                  n_s=0.9667, sigma8=0.8159, Omega_k=0.0, w0=-1.0, wa=0.0)
    a_grid = np.linspace(0.2, 0.8, 50)

    # 2. Run the computation (automatically tracked by @track)
    distances = comoving_angular_distance(params, a_grid)

    # 3. Generate training data over a parameter grid
    rng = np.random.default_rng(42)
    samples = [
        {**params, "Omega_c": rng.uniform(0.22, 0.32), "h": rng.uniform(0.62, 0.75),
         "sigma8": rng.uniform(0.77, 0.87)}
        for _ in range(50)
    ]
    param_names = ["Omega_c", "h", "sigma8"]
    X, y = build_training_data(comoving_angular_distance, samples, a_grid, param_names=param_names)

    # 4. Train a GP emulator
    emu = GPEmulator(feature_names=param_names + ["a"])
    emu.fit(X, y)

    # 5. Hot-swap: replace pyccl with the emulator
    compute = swappable(comoving_angular_distance)
    register(comoving_angular_distance, emu, param_names)
    result = compute(params, a_grid)  # uses emulator — 40x faster
