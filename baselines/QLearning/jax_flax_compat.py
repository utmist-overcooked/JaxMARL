"""Compat shim: flax 0.10.4 calls a newer jax `debug_info` API than jax 0.4.38 has.

`flax/core/axes_scan.py` does:

    debug_info = jax.api_util.debug_info("flax scan", body, (in_tree,), {})
    f_flat, out_tree = jax.api_util.flatten_fun_nokwargs(
        lu.wrap_init(body, debug_info=debug_info), in_tree)

but jax 0.4.38 has neither `jax.api_util.debug_info` nor a `debug_info` kwarg on
`linear_util.wrap_init`. Any `nn.scan` whose body touches `setup()`-defined submodules
hits this path and dies with:

    AttributeError: module 'jax.api_util' has no attribute 'debug_info'

(transf_qmix's ScannedTransformer is one; QMIX's ScannedRNN happens to avoid it.)

`debug_info` only enriches error messages — it carries no numerics — so back-filling it
with a no-op is safe. Import this module before any `nn.scan` is traced.
"""

import inspect

import jax.api_util as _api_util
from jax.extend import linear_util as _lu

if not hasattr(_api_util, "debug_info"):
    _api_util.debug_info = lambda *args, **kwargs: None

if "debug_info" not in inspect.signature(_lu.wrap_init).parameters:
    _orig_wrap_init = _lu.wrap_init

    def _wrap_init(f, params=None, debug_info=None, **kwargs):
        return _orig_wrap_init(f, params)

    _lu.wrap_init = _wrap_init
