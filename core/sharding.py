from __future__ import annotations

from typing import TypeVar

import flax.nnx as nnx
import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

_T = TypeVar("_T")

# Axis names used by the two parallelism dimensions
DATA_AXIS = "data"
TP_AXIS   = "model"


def make_mesh(n_devices: int = 0) -> Mesh:
    """Create a 1-D data-parallel mesh.

    Parameters
    ----------
    n_devices:
        Number of devices to use.  0 (default) means all available local devices.
    """
    devices = jax.local_devices()
    if 0 < n_devices < len(devices):
        devices = devices[:n_devices]
    return Mesh(np.array(devices), axis_names=(DATA_AXIS,))


def make_tp_mesh(n_tp: int, n_dp: int = 1) -> Mesh:
    """Create a 2-D (data × model) mesh for combined data + tensor parallelism.

    Parameters
    ----------
    n_tp:
        Number of devices on the tensor-parallel (model) axis.
    n_dp:
        Number of devices on the data-parallel axis.  Defaults to 1 (pure TP).
        The total devices used is ``n_dp * n_tp``.
    """
    devices = jax.local_devices()[: n_dp * n_tp]
    return Mesh(
        np.array(devices).reshape(n_dp, n_tp),
        axis_names=(DATA_AXIS, TP_AXIS),
    )


def replicate(pytree: _T, mesh: Mesh) -> _T:
    """Copy *pytree* to every device in *mesh* (no sharding on any axis)."""
    sharding = NamedSharding(mesh, P())
    return jax.device_put(pytree, sharding)


def shard_batch(pytree: _T, mesh: Mesh) -> _T:
    """Shard *pytree* along its leading (batch) axis across all devices in *mesh*."""
    sharding = NamedSharding(mesh, P(DATA_AXIS))
    return jax.device_put(pytree, sharding)


def num_devices(mesh: Mesh) -> int:
    """Return the total number of devices in *mesh*."""
    return mesh.size


def apply_tp_sharding(model: nnx.Module, mesh: Mesh) -> None:
    """Shard model weights in-place for Megatron-style tensor parallelism.

    Column-parallel layers (qkv, up_proj): output axis sharded on TP_AXIS.
    Row-parallel layers (o_proj, down_proj): input axis sharded on TP_AXIS.

    Row-parallel biases are scaled by ``1 / n_tp`` so that the all-reduce
    (sum) inside the forward pass restores the original bias value.

    Must be called before the first JIT-compiled forward pass.
    """
    n_tp = mesh.shape[TP_AXIS]

    def _put(arr, spec: P) -> jax.Array:
        return jax.device_put(arr, NamedSharding(mesh, spec))

    # Column-parallel: output (last) axis sharded
    COL_PARALLEL = {"qkv", "up_proj"}
    # Row-parallel: input (first) axis sharded
    ROW_PARALLEL = {"o_proj", "down_proj"}

    state = nnx.state(model)

    def _shard(path, leaf):
        if leaf is None or not isinstance(leaf, jax.Array):
            return leaf

        # Build a set of path-part strings for matching
        parts: set[str] = set()
        for p in path:
            parts.add(str(p.key) if hasattr(p, "key") else str(p))

        is_kernel = "kernel" in parts
        is_bias   = "bias"   in parts
        is_col    = bool(parts & COL_PARALLEL)
        is_row    = bool(parts & ROW_PARALLEL)

        if is_kernel and is_col and leaf.ndim == 2:
            return _put(leaf, P(None, TP_AXIS))

        if is_kernel and is_row and leaf.ndim == 2:
            return _put(leaf, P(TP_AXIS, None))

        if is_bias and is_col and leaf.ndim == 1:
            # Each device holds its slice of the column-parallel bias
            return _put(leaf, P(TP_AXIS))

        if is_bias and is_row:
            # Scale by 1/n_tp so the post-all-reduce sum equals the original bias
            return _put(leaf / n_tp, P())

        return _put(leaf, P())

    new_state = jax.tree_util.tree_map_with_path(_shard, state)
    nnx.update(model, new_state)
