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

def get_batch(data: jnp.ndarray, batch_size: int, max_context: int, key: jax.Array):
    ix = jax.random.randint(key, (batch_size,), 0, len(data) - max_context)
    x = jnp.stack([data[i:i+max_context] for i in ix])
    y = jnp.stack([data[i+1:i+max_context+1] for i in ix])
    return x, y

def lr_schedule(step: int, base_lr: float, warmup_steps: int, total_steps: int):
    def lr_fn(step):
        is_warmup = step < warmup_steps
        warmup_lr = base_lr * (step / jnp.maximum(1, warmup_steps))

        progress = (step - warmup_steps) / jnp.maximum(1, total_steps - warmup_steps)
        cosine_lr = 0.5 * base_lr * (1 + jnp.cos(jnp.pi * progress))

        return jnp.where(is_warmup, warmup_lr, cosine_lr)
    return lr_fn(step)
