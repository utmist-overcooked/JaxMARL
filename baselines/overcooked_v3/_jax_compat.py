"""Compatibility shims for the JAX/Flax versions supported by this checkout."""

import inspect

import jax
import jax.api_util
from jax.extend import linear_util as jax_linear_util


if not hasattr(jax.interpreters.xla, "pytype_aval_mappings"):
    jax.interpreters.xla.pytype_aval_mappings = jax.core.pytype_aval_mappings

if not hasattr(jax.api_util, "debug_info"):
    jax.api_util.debug_info = lambda *args, **kwargs: None

if "debug_info" not in inspect.signature(jax_linear_util.wrap_init).parameters:
    _jax_wrap_init = jax_linear_util.wrap_init

    def _wrap_init_compat(f, params=None, *, debug_info=None):
        """Ignore the newer debug argument on older JAX versions."""

        del debug_info
        return _jax_wrap_init(f, params=params)

    jax_linear_util.wrap_init = _wrap_init_compat
