"""Finite Scalar Quantization utilities for JAX.

This is a small, import-safe adaptation of the official Google Research FSQ
JAX implementation:
https://github.com/google-research/google-research/tree/master/fsq
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

Codeword = jax.Array
Indices = jax.Array


def round_ste(z: jax.Array) -> jax.Array:
    """Round with straight-through gradients."""
    zhat = jnp.round(z)
    return z + jax.lax.stop_gradient(zhat - z)


@dataclass(frozen=True)
class FSQ:
    """Finite Scalar Quantizer.

    Args:
        levels: Number of scalar levels for each code dimension.
        eps: Small bound shrinkage used by the original implementation.
    """

    levels: Sequence[int]
    eps: float = 1e-3

    def __post_init__(self):
        if not self.levels:
            raise ValueError("FSQ levels must not be empty.")
        if any(level < 2 for level in self.levels):
            raise ValueError("Each FSQ level must be at least 2.")

        object.__setattr__(self, "_levels_np", np.asarray(self.levels, dtype=np.int32))
        basis = np.concatenate(([1], np.cumprod(self._levels_np[:-1]))).astype(
            np.uint32
        )
        object.__setattr__(self, "_basis_np", basis)

    @property
    def num_dimensions(self) -> int:
        return len(self.levels)

    @property
    def codebook_size(self) -> int:
        return int(np.prod(self._levels_np))

    @property
    def codebook(self) -> Codeword:
        return self.indexes_to_codes(jnp.arange(self.codebook_size, dtype=jnp.uint32))

    @property
    def levels_array(self) -> jax.Array:
        return jnp.asarray(self._levels_np)

    @property
    def basis_array(self) -> jax.Array:
        return jnp.asarray(self._basis_np)

    def bound(self, z: jax.Array) -> jax.Array:
        """Bound input values before scalar rounding."""
        levels = self.levels_array
        half_l = (levels - 1) * (1.0 - self.eps) / 2.0
        offset = jnp.where(levels % 2 == 1, 0.0, 0.5)
        shift = jnp.tan(offset / half_l)
        return jnp.tanh(z + shift) * half_l - offset

    def quantize(self, z: jax.Array) -> Codeword:
        """Quantize `z` and renormalize codes to approximately [-1, 1]."""
        if z.shape[-1] != self.num_dimensions:
            raise ValueError(
                f"Expected last dim {self.num_dimensions}, got {z.shape[-1]}."
            )

        quantized = round_ste(self.bound(z))
        half_width = self.levels_array // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: jax.Array) -> jax.Array:
        half_width = self.levels_array // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: jax.Array) -> jax.Array:
        half_width = self.levels_array // 2
        return (zhat - half_width) / half_width

    def codes_to_indexes(self, zhat: Codeword) -> Indices:
        """Convert normalized code vectors to integer codebook indexes."""
        if zhat.shape[-1] != self.num_dimensions:
            raise ValueError(
                f"Expected last dim {self.num_dimensions}, got {zhat.shape[-1]}."
            )

        zhat = self._scale_and_shift(zhat)
        return (zhat * self.basis_array).sum(axis=-1).astype(jnp.uint32)

    def indexes_to_codes(self, indices: Indices) -> Codeword:
        """Convert integer codebook indexes to normalized code vectors."""
        indices = indices[..., jnp.newaxis]
        codes_non_centered = jnp.mod(
            jnp.floor_divide(indices, self.basis_array), self.levels_array
        )
        return self._scale_and_shift_inverse(codes_non_centered)

    def quantize_and_index(self, z: jax.Array) -> tuple[Codeword, Indices]:
        codes = self.quantize(z)
        return codes, self.codes_to_indexes(codes)
