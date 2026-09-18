Emulator Inversion
==================

Given target outputs (e.g., observed distances), recover the cosmological
parameters that produced them.

Basic Usage
-----------

::

    result = emu.invert(
        y_target=observed_distances,
        free_params=["Omega_c", "h"],
        fixed_params={"sigma8": 0.8159, "a": a_grid},
        x0={"Omega_c": 0.27, "h": 0.68},
        bounds={"Omega_c": (0.2, 0.35), "h": (0.6, 0.8)},
    )
    print(result.x_solution)   # {"Omega_c": 0.259, "h": 0.677}
    print(result.residual)     # relative fit quality

Backend-Specific Methods
------------------------

Each emulator backend provides a specialized inversion:

.. list-table::
   :header-rows: 1

   * - Backend
     - Method
     - Advantage
   * - Default
     - ``scipy.optimize.minimize`` (L-BFGS-B)
     - Works with all backends
   * - GP
     - Uncertainty-weighted ``(pred-target)^2/std^2``
     - Steers toward well-explored regions
   * - PyTorch
     - ``torch.autograd`` + Adam
     - Gradient-based, fast convergence
   * - TensorFlow
     - ``tf.GradientTape`` + Adam
     - Same, TF ecosystem
   * - PySR
     - ``sympy.solve`` for single free param
     - Exact algebraic solution when possible

InversionResult
---------------

All inversion methods return an ``InversionResult``::

    @dataclass
    class InversionResult:
        x_solution: dict[str, float]   # recovered parameter values
        y_predicted: np.ndarray        # emulator prediction at solution
        y_target: np.ndarray           # the target values
        residual: float                # relative residual
        success: bool
        message: str
