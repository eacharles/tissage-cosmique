tissage-cosmique
================

Cosmological computations with provenance tracking and emulator support.

**tissage-cosmique** wraps `pyccl <https://github.com/LSSTDESC/CCL>`_ (the Core
Cosmology Library) to run cosmological computations, store inputs and outputs
with automatic provenance tracking via
`tisserande <https://github.com/eacharles/tisserande>`_, and build emulators
(fast surrogate models) that can transparently replace the original code.

Built on:

- **macon** — layered CRUD database framework
- **tisserande** — execution provenance tracking
- **pyccl** — Core Cosmology Library

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
