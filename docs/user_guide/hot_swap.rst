Hot-Swap
========

The swap module transparently routes computation calls through a trained
emulator instead of pyccl, with no changes to calling code.

Basic Usage
-----------

::

    from tissage_cosmique.swap import swappable, register, original

    # 1. Wrap a computation to make it swappable
    compute = swappable(comoving_angular_distance)

    # 2. Register a trained emulator
    register(comoving_angular_distance, emu, param_names=["Omega_c", "h", "sigma8"])

    # 3. Calls now route through the emulator
    result = compute(params, a)  # uses emulator

    # 4. Temporarily use the original pyccl function
    with original(comoving_angular_distance):
        truth = compute(params, a)  # uses pyccl

Registry Control
----------------

::

    from tissage_cosmique.swap import (
        register, unregister, enable, disable,
        enable_all, disable_all, is_registered, list_registered, reset,
    )

    register(fn, emulator, param_names)   # add an emulator
    unregister(fn)                        # remove it
    enable(fn) / disable(fn)              # toggle one function
    enable_all() / disable_all()          # toggle all
    is_registered(fn)                     # check
    list_registered()                     # {qualname: SwapEntry}
    reset()                               # clear everything

How It Works
------------

1. ``swappable(fn)`` returns a wrapper that checks the global registry on each call
2. If ``fn`` has a registered, enabled, fitted emulator:

   - Convert ``(cosmo_params_dict, a_array)`` → feature matrix via
     ``params_to_feature_matrix``
   - Call ``emulator.predict(X)``
   - Return the result

3. Otherwise, call the original function normally

The ``original(fn)`` context manager temporarily sets ``entry.enabled = False``
and restores it on exit, ensuring the emulator is bypassed for validation or
ground-truth comparison.
