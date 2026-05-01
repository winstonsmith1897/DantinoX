from __future__ import annotations

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


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
    return Mesh(np.array(devices), axis_names=("data",))


def replicate(pytree: object, mesh: Mesh) -> object:
    """Copy *pytree* to every device in *mesh* (no sharding on any axis)."""
    sharding = NamedSharding(mesh, P())
    return jax.device_put(pytree, sharding)


def shard_batch(pytree: object, mesh: Mesh) -> object:
    """Shard *pytree* along its leading (batch) axis across all devices in *mesh*."""
    sharding = NamedSharding(mesh, P("data"))
    return jax.device_put(pytree, sharding)


def num_devices(mesh: Mesh) -> int:
    """Return the total number of devices in *mesh*."""
    return mesh.size
