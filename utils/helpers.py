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
    x = jnp.stack([data[i:i+max_context] for i in ix])
    y = jnp.stack([data[i+1:i+max_context+1] for i in ix])
    return x, y

