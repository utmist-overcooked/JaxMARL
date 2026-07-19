import itertools
import jax
import jax.numpy as jnp
import numpy as np

Codeword = jax.Array
Indices = jax.Array


def round_ste(z):
  """Round with straight through gradients."""
  zhat = jnp.round(z)
  return z + jax.lax.stop_gradient(zhat - z)


class FSQ:
  """Quantizer."""

  def __init__(self, levels: list[int], eps: float = 1e-3):
    self._levels = levels
    self._eps = eps
    self._levels_np = np.asarray(levels)
    self._basis = np.concatenate(
        ([1], np.cumprod(self._levels_np[:-1]))).astype(np.uint32)

    self._implicit_codebook = self.indexes_to_codes(
        np.arange(self.codebook_size))

  @property
  def num_dimensions(self) -> int:
    """Number of dimensions expected from inputs."""
    return len(self._levels)

  @property
  def codebook_size(self) -> int:
    """Size of the codebook."""
    return np.prod(self._levels)

  @property
  def codebook(self):
    """Returns the implicit codebook. Shape (prod(levels), num_dimensions)."""
    return self._implicit_codebook

  def bound(self, z: jax.Array) -> jax.Array:
    """Bound `z`, an array of shape (..., d)."""
    half_l = (self._levels_np - 1) * (1 - self._eps) / 2
    offset = jnp.where(self._levels_np % 2 == 1, 0.0, 0.5)
    shift = jnp.tan(offset / half_l)
    return jnp.tanh(z + shift) * half_l - offset

  def quantize(self, z: jax.Array) -> Codeword:
    """Quanitzes z, returns quantized zhat, same shape as z."""
    quantized = round_ste(self.bound(z))

    # Renormalize to [-1, 1].
    half_width = self._levels_np // 2
    return quantized / half_width

  def _scale_and_shift(self, zhat_normalized):
    # Scale and shift to range [0, ..., L-1]
    half_width = self._levels_np // 2
    return (zhat_normalized * half_width) + half_width

  def _scale_and_shift_inverse(self, zhat):
    half_width = self._levels_np // 2
    return (zhat - half_width) / half_width

  def codes_to_indexes(self, zhat: Codeword) -> Indices:
    """Converts a `code` to an index in the codebook."""
    assert zhat.shape[-1] == self.num_dimensions
    zhat = self._scale_and_shift(zhat)
    return (zhat * self._basis).sum(axis=-1).astype(jnp.uint32)

  def indexes_to_codes(self, indices: Indices) -> Codeword:
    """Inverse of `indexes_to_codes`."""
    indices = indices[..., jnp.newaxis]
    codes_non_centered = np.mod(
        np.floor_divide(indices, self._basis), self._levels_np
    )
    return self._scale_and_shift_inverse(codes_non_centered)


if __name__ == "__main__":
  fsq = FSQ(levels=[3, 5, 4])

  z = np.asarray([0.25, 0.6, -7])
  zhat = fsq.quantize(z)
  print(f"Quantized {z} -> {zhat}")

  idx = fsq.codes_to_indexes(zhat)
  print(f"Code {zhat} is the {idx}-th index.")

  code_out = fsq.indexes_to_codes(idx)
  print(f"Index {idx} mapped back to {code_out}.")

  fsq = FSQ(levels=[5, 4, 3])

  d = fsq.num_dimensions
  z = np.random.uniform(size=(3, 8, 8, d))
  zhat = fsq.quantize(z)
  assert zhat.shape == (3, 8, 8, d)

  indices = fsq.codes_to_indexes(zhat)
  assert indices.shape == (3, 8, 8)

  zhat_out = fsq.indexes_to_codes(indices)
  assert zhat_out.shape == zhat.shape

  np.testing.assert_allclose(zhat, zhat_out)

