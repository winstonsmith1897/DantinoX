import argparse
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import time
import csv
import os
import json
import datetime
import math
from datasets import load_dataset
from core.config import Config
from core.model import Transformer
from utils.tokenizer import get_tokenizer
from utils.helpers import compute_loss, get_batch
import dataclasses
import wandb

def get_optax_optimizer(config, total_steps):
    requested_warmup = getattr(config, 'warmup_steps', 0)
    warmup_steps = min(requested_warmup, int(total_steps * 0.3))
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.lr,
        warmup_steps=warmup_steps,
        decay_steps=max(total_steps, warmup_steps + 1),
        end_value=config.lr * 0.01  
    )
    opt_name = config.optimizer.lower()
    if opt_name == "adamw": return optax.adamw(learning_rate=lr_schedule)
    if opt_name == "lion": return optax.lion(learning_rate=lr_schedule)
    return optax.adam(learning_rate=lr_schedule)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--wandb_project", type=str, default="DantinoX")
    for field in dataclasses.fields(Config):
        ftype = field.type
        arg_name = f"--{field.name}"
        if arg_name not in parser._option_string_actions:
            parser.add_argument(arg_name, type=ftype)
    args, _ = parser.parse_known_args()
    return args

def get_vram_usage():
    devices = jax.devices()
    try:
        for d in devices:
            if d.platform == 'gpu':
                stats = d.memory_stats()
                return stats['bytes_in_use'] / 1e9
    except: return 0.0
    return 0.0

def report_model_summary(model, config, optimizer, save_path):
    params = nnx.state(model, nnx.Param)
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    weights_mem = total_params * 4 / 1e6
    opt_state = nnx.state(optimizer)
    opt_params = sum(x.size for x in jax.tree_util.tree_leaves(opt_state) if isinstance(x, jax.Array))
    opt_mem = opt_params * 4 / 1e6
    act_mem = (config.batch_size * config.max_context * config.dim * config.num_blocks * 8 * 4) / 1e6
    summary = {
        "total_params_M": round(total_params / 1e6, 2),
        "weights_mem_MB": round(weights_mem, 2),
        "optimizer_mem_MB": round(opt_mem, 2),
        "est_activations_MB": round(act_mem, 2),
        "total_est_vram_MB": round(weights_mem + opt_mem + act_mem, 2)
    }
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=4)
    return summary

def main():
    wandb.init(group="Attention-Comparison-V1")
    
    args = parse_args()
    config = Config.from_yaml(args.config)
    
    for k, v in wandb.config.items():
        if hasattr(config, k):
            setattr(config, k, v)
            
    if getattr(wandb.config, 'activation', 'silu') == "gelu":
        config.use_swiglu = False
        config.activation = "gelu"
    else:
        config.use_swiglu = getattr(wandb.config, 'use_swiglu', True)
        config.activation = "silu"

    config.head_size = 32
    config.n_heads = config.dim // config.head_size
    attn_type = wandb.config.get("attention_type", "standard_mha")
    
    if attn_type == "standard_mha":
        config.mla = False
        config.kv_heads = config.n_heads
    elif attn_type == "standard_gqa":
        config.mla = False
        config.kv_heads = max(1, config.n_heads // 4)
    elif attn_type == "mla":
        config.mla = True
        config.kv_heads = max(1, config.n_heads // 4)
        config.down_dim_q = wandb.config.get("down_dim_q", config.dim // 2)
        config.down_dim_kv = wandb.config.get("down_dim_kv", config.dim // 4)
        config.rope_dim = wandb.config.get("rope_dim", 16)

    if config.n_heads % config.kv_heads != 0:
        config.kv_heads = math.gcd(config.n_heads, config.kv_heads)

    if config.mla:
        kv_bytes = 2 * (config.down_dim_kv + config.rope_dim)
    else:
        kv_bytes = 2 * 2 * (config.kv_heads * config.head_size)
    
    config.use_rotary_pos = True
    config.absolute_pos = False
    config.trainable_pos = False
    
    moe_tag = "MoE" if config.use_moe else "Dense"
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    run_id = f"{attn_type}_{config.dim}d_{config.num_blocks}b_{moe_tag}_{timestamp}"
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    metadata = {
        **dataclasses.asdict(config),
        "attn_type": attn_type,
        "kv_bytes_per_token": kv_bytes,
        "run_id": run_id
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    wandb.config.update(metadata, allow_val_change=True)
    config.save_yaml(os.path.join(run_dir, "config.yaml"))

    if config.dataset_source == "huggingface":
        raw_dataset = load_dataset(config.dataset_name, split='train')
        text = " ".join(raw_dataset['text'])
    else:
        path = args.data_path if args.data_path else config.dataset_name
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

    raw_lines = text.split('\n')
    valid_lines = [l.rstrip() for l in raw_lines if l.strip()]
    formatted_blocks = []
    for i in range(0, len(valid_lines), 3):
        formatted_blocks.append('\n'.join(valid_lines[i:i+3]))
    text = '\n\n'.join(formatted_blocks) + '\n'

    tokenizer = get_tokenizer(config.tokenizer_type)
    if config.tokenizer_type == "char":
        tokenizer.train_from_text(text) 
    elif config.tokenizer_type == "bpe":
        tokenizer.train_from_text(text, vocab_size=config.vocab_size)
    
    config.vocab_size = tokenizer.vocab_size
    full_data = jnp.array(tokenizer.encode(text), dtype=jnp.int32)
    n = int(0.9 * len(full_data))
    train_data, val_data = full_data[:n], full_data[n:]

    total_steps = (len(train_data) // (config.batch_size * config.max_context)) * config.epochs
    rngs = nnx.Rngs(config.seed)
    model = Transformer(config, rngs=rngs)
    tx = get_optax_optimizer(config, total_steps)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    
    report_model_summary(model, config, optimizer, os.path.join(run_dir, "model_summary.json"))
    
    log_f = open(os.path.join(run_dir, "training_log.csv"), 'a', newline='')
    log_writer = csv.writer(log_f)
    log_writer.writerow(['step', 'train_loss', 'val_loss', 'vram_gb', 'ms_per_step'])

    micro_batch_size = config.batch_size // config.grad_accum
    
    def loss_fn(model, x, y):
        logits, _, bal_loss = model(x, use_cache=False, kv_caches=None, cache_index=0)
        loss = compute_loss(logits, y)
        if getattr(model, 'use_moe', False):
            loss += config.alpha_balance * bal_loss
        return loss, bal_loss

    @jax.jit
    def train_step(graphdef, state, full_x, full_y):
        model, optimizer, metrics = nnx.merge(graphdef, state)
        x_b = full_x.reshape(config.grad_accum, micro_batch_size, -1)
        y_b = full_y.reshape(config.grad_accum, micro_batch_size, -1)
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        grad_acc = jax.tree_util.tree_map(jnp.zeros_like, nnx.state(model, nnx.Param))
        total_l, total_b = 0.0, 0.0
        for i in range(config.grad_accum):
            (l, b), g = grad_fn(model, x_b[i], y_b[i])
            grad_acc = jax.tree_util.tree_map(lambda acc, grad: acc + grad / config.grad_accum, grad_acc, g)
            total_l += l / config.grad_accum
            total_b += b / config.grad_accum
        optimizer.update(model, grad_acc)
        metrics.update(loss=total_l)
        return total_l, total_b, nnx.state((model, optimizer, metrics))

    @nnx.jit
    def eval_step(model, x, y):
        return loss_fn(model, x, y)

    metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))
    key = jax.random.PRNGKey(config.seed)
    t0 = time.time()

    try:
        for step in range(total_steps):
            key, subkey = jax.random.split(key)
            x, y = get_batch(train_data, config.batch_size, config.max_context, subkey)
            graphdef, state = nnx.split((model, optimizer, metrics))
            l, bl, new_state = train_step(graphdef, state, x, y)
            nnx.update((model, optimizer, metrics), new_state)
            
            if step % 50 == 0:
                dt = (time.time() - t0) * 1000 / 50 if step > 0 else 0
                t0 = time.time()
                vram = get_vram_usage()
                
                key, subkey = jax.random.split(key)
                vx, vy = get_batch(val_data, micro_batch_size, config.max_context, subkey)
                vl, vb = eval_step(model, vx, vy)
                
                wandb.log({
                    "step": step, "train_loss": float(l), "val_loss": float(vl),
                    "vram_gb": vram, "ms_per_step": dt, "kv_bytes": kv_bytes
                })
                log_writer.writerow([step, float(l), float(vl), round(vram, 3), round(dt, 2)])
                log_f.flush()
        
        final_params = nnx.state(model, nnx.Param).to_pure_dict()
        with open(os.path.join(run_dir, "model_weights.msgpack"), "wb") as f:
            import flax.serialization
            f.write(flax.serialization.msgpack_serialize(final_params))
            
    finally:
        log_f.close()
        wandb.finish()

if __name__ == "__main__":
    main()