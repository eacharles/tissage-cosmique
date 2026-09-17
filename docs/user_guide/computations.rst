Computations
============

Computation functions wrap pyccl calls and are decorated with tisserande's
``@track`` for automatic provenance capture.

Wrapping a pyccl Function
--------------------------

Each computation function follows this pattern:

1. Takes a parameter dictionary (``ConfigDict[dict]``) and one or more grid
   arrays (``ArrayArg[np.ndarray]``)
2. Converts the dict to a ``pyccl.Cosmology`` via ``make_pyccl_cosmology``
3. Calls the pyccl function
4. Returns the result as a numpy array

Example — comoving angular distance::

    from tissage_cosmique.computations.distances import comoving_angular_distance

    params = dict(Omega_c=0.2589, Omega_b=0.0486, h=0.6774,
                  n_s=0.9667, sigma8=0.8159)
    a = np.linspace(0.2, 1.0, 100)
    distances = comoving_angular_distance(params, a)

Example — linear matter power spectrum::

    from tissage_cosmique.computations.power import linear_matter_power

    k = np.logspace(-3, 0, 50)
    a = np.array([0.5, 0.8, 1.0])
    pk = linear_matter_power(params, k, a)  # shape (n_a, n_k)

The ``make_pyccl_cosmology`` Helper
------------------------------------

Converts a parameter dictionary to a ``pyccl.Cosmology`` object. Strips
non-pyccl keys (``name``, ``id_``) and drops ``None`` values so that the
``sigma8``/``A_s`` mutual exclusivity is respected::

    from tissage_cosmique.computations.cosmology import make_pyccl_cosmology

    cosmo = make_pyccl_cosmology(params)

Provenance Tracking
-------------------

When tisserande tracking is configured, every call to a ``@track``-decorated
function records:

- An **execution** record (status, timing, errors)
- **Input nodes** (cosmological parameters as ``config_dict``, grid arrays)
- **Output nodes** (result arrays)
- **Edges** connecting inputs → function → outputs

Without a configured backend, ``@track`` is transparent — the function runs
normally with no overhead.
