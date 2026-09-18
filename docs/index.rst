tissage-cosmique
================

.. image:: _static/Tissage-Cosmique-logo.jpg
   :align: center
   :width: 400px

|

Cosmological computations with provenance tracking, emulator support, and
hot-swap dispatch.

**tissage-cosmique** wraps `pyccl <https://github.com/LSSTDESC/CCL>`_ (the Core
Cosmology Library) to:

1. **Run cosmological computations** (distances, power spectra) with automatic
   provenance tracking via `tisserande <https://github.com/eacharles/tisserande>`_
2. **Store inputs, outputs, and provenance** in a database via macon's CRUD layer
3. **Build emulators** (GP, TensorFlow, PyTorch, PySR) trained on stored results
4. **Hot-swap emulators** for the original code transparently at runtime

Built on:

- `macon <https://github.com/eacharles/macon>`_ — layered CRUD database framework
- `tisserande <https://github.com/eacharles/tisserande>`_ — execution provenance tracking
- `pyccl <https://github.com/LSSTDESC/CCL>`_ — Core Cosmology Library

Features
--------

- **4 emulator backends**: Gaussian Process, TensorFlow, PyTorch, PySR (symbolic regression)
- **Latent space emulation**: PCA and autoencoder codecs for compressed output representation
- **Emulator inversion**: recover input parameters from target outputs
- **Validation tools**: accuracy metrics, cross-validation, uncertainty calibration
- **DB tracking**: EmulatorRecord and CodecRecord tables linked to provenance
- **Hot-swap dispatch**: ``swappable()`` wrapper + ``original()`` context manager
- **Dashboard**: interactive Plotly/Dash web interface
- **Full macon integration**: CLI, FastAPI router, async/sync operations

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting_started
   user_guide/index
   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
