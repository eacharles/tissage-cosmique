Latent Space Emulation
======================

Instead of emulating each grid point independently, compress the full output
curve into a low-dimensional latent code and emulate that.

Codecs
------

Two codec implementations:

- **PCACodec**: Linear compression via PCA. For smooth cosmological functions,
  3 components typically capture >99.99% of variance.
- **AutoencoderCodec**: Nonlinear compression via a PyTorch encoder/decoder
  network. More expressive for functions with sharp features.

::

    from tissage_cosmique.emulators import PCACodec

    codec = PCACodec(n_components=3)
    codec.fit(Y_curves)              # Y: (n_samples, n_grid)
    Z = codec.encode(Y_curves)       # Z: (n_samples, 3)
    Y_recon = codec.decode(Z)        # reconstructed curves

LatentEmulator
--------------

Wraps a codec and K independent scalar emulators (one per latent dimension)::

    from tissage_cosmique.emulators import LatentEmulator, PCACodec, GPEmulator

    le = LatentEmulator(
        codec=PCACodec(n_components=3),
        emulator_factory=lambda: GPEmulator(feature_names=param_names),
    )
    le.fit(param_samples, computation_fn, a_grid, param_names=param_names)
    curve = le.predict(cosmo_params)      # full output curve
    z = le.predict_latent(cosmo_params)   # latent code only

DB Tracking
-----------

Fitted codecs and trained emulators can be registered in the database via
``CodecRecord`` and ``EmulatorRecord`` tables, linking them to the tisserande
provenance executions they were trained on.
