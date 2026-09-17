Emulators
=========

Emulators are surrogate models trained to reproduce cosmological computations
at a fraction of the cost.

The Emulator Interface
----------------------

All backends implement the same abstract interface
(:class:`~tissage_cosmique.emulators.base.Emulator`):

- ``fit(X, y)`` — train on a feature matrix and target array
- ``predict(X)`` — predict targets for new inputs
- ``save(path)`` / ``load(path)`` — serialize to/from disk
- ``metadata`` — dict with type, training info, backend-specific details

Available Backends
------------------

.. list-table::
   :header-rows: 1

   * - Backend
     - Class
     - Strengths
   * - Gaussian Process
     - :class:`~tissage_cosmique.emulators.backends.gp.GPEmulator`
     - Best accuracy at small N, uncertainty estimates
   * - TensorFlow
     - :class:`~tissage_cosmique.emulators.backends.tensorflow.TensorFlowEmulator`
     - Scales to large N, fast batch prediction
   * - PyTorch
     - :class:`~tissage_cosmique.emulators.backends.pytorch.PyTorchEmulator`
     - Same as TF, Pythonic API
   * - Symbolic (PySR)
     - :class:`~tissage_cosmique.emulators.backends.symbolic.SymbolicEmulator`
     - Discovers closed-form expressions, zero runtime deps

Building Training Data
----------------------

The ``build_training_data`` function runs a computation over a parameter grid
and flattens the results into ``(X, y)`` pairs::

    from tissage_cosmique.emulators import build_training_data

    # Single grid (distances)
    X, y = build_training_data(
        comoving_angular_distance, samples, a_grid,
        param_names=["Omega_c", "h", "sigma8"],
    )

    # Two grids (power spectrum) — specify axis mapping
    X, y = build_training_data(
        linear_matter_power, samples, k_grid, a_grid,
        param_names=["Omega_c", "h", "sigma8"],
        result_axes_order=[1, 0],  # result shape (n_a, n_k)
    )

Training from the Database
--------------------------

Instead of re-running pyccl, you can query previously stored provenance::

    from tissage_cosmique.emulators import query_computation_results

    X, y = query_computation_results(param_names=["Omega_c", "h", "sigma8"])

Validation
----------

Accuracy metrics, cross-validation, and uncertainty calibration::

    from tissage_cosmique.emulators import (
        validate_emulator,
        validate_against_computation,
        cross_validate,
        check_calibration,
    )

    result = validate_against_computation(emu, computation_fn, test_params, a_grid, param_names)
    print(result.r2_score, result.mean_relative_error)

    cal = check_calibration(emu, X_test, y_test)
    print(cal.expected_coverage, cal.observed_coverage)
