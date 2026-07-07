import jax
import jax.numpy as jnp


def compute_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    B, T, C = logits.shape
    logits = jnp.reshape(logits, (B * T, C))
    targets = jnp.reshape(targets, (B * T))
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.take_along_axis(
        log_probs,
        targets[:, None],
        axis=-1
    ).squeeze(-1)
    return loss.mean()

def get_batch(
    data: jnp.ndarray, batch_size: int, max_context: int, key: jax.Array
) -> tuple[jnp.ndarray, jnp.ndarray]:
    ix = jax.random.randint(key, (batch_size,), 0, len(data) - max_context)
    # Vectorized gather: a single XLA gather op instead of 2*batch_size
    # separate dynamic-slice dispatches (the previous Python-loop version
    # left the GPU idle between many small, serially-dispatched slices).
    offsets = jnp.arange(max_context)
    gather_idx = ix[:, None] + offsets[None, :]  # (batch_size, max_context)
    x = data[gather_idx]
    y = data[gather_idx + 1]
    return x, y

