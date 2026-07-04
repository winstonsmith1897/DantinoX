"""Minimal 2-GPU tensor-parallel sanity test: sharded matmul + all-gather."""
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

mesh = Mesh(np.array(jax.devices()[:2]).reshape(1, 2), ("data", "model"))
x = jax.device_put(jnp.ones((8, 4096)), NamedSharding(mesh, P()))
w = jax.device_put(jnp.ones((4096, 4096)), NamedSharding(mesh, P(None, "model")))


@jax.jit
def f(x, w):
    return jax.lax.with_sharding_constraint(x @ w, NamedSharding(mesh, P()))


jax.block_until_ready(f(x, w))
print("TP-OK")
