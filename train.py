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
from datasets import load_dataset
from core.config import Config
from core.model import Transformer
from utils.tokenizer import get_tokenizer
from utils.helpers import compute_loss, get_batch
import dataclasses

def get_optax_optimizer(name, lr):
    try:
        opt_func = getattr(optax, name.lower())
        return opt_func(learning_rate=lr)
    except AttributeError:
        return optax.adamw(learning_rate=lr)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--data_path", type=str)
    
    for field in dataclasses.fields(Config):
        ftype = field.type
        arg_name = f"--{field.name}"
        if arg_name not in parser._option_string_actions:
            parser.add_argument(arg_name, type=ftype)
            
    return parser.parse_args()

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
    print(f"Params: {summary['total_params_M']}M | Est. VRAM: {summary['total_est_vram_MB']}MB")

def main():
    args = parse_args()
    config = Config.from_yaml(args.config)
    args_dict = vars(args)
    for field in dataclasses.fields(Config):
        val = args_dict.get(field.name)
        if val is not None: setattr(config, field.name, val)
    
    run_name = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    config.save_yaml(os.path.join(run_dir, "config.yaml"))
    
    log_file_path = os.path.join(run_dir, "training_log.csv")
    summary_file_path = os.path.join(run_dir, "model_summary.json")

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

    tokenizer = get_tokenizer(config.tokenizer_type, text=text if config.tokenizer_type == "char" else None)
    if config.tokenizer_type == "bpe":
        tokenizer.train_from_text(text, vocab_size=config.vocab_size)
    
    config.vocab_size = tokenizer.vocab_size
    full_data = jnp.array(tokenizer.encode(text), dtype=jnp.int32)
    n = int(0.9 * len(full_data))
    train_data, val_data = full_data[:n], full_data[n:]

    tokens_per_step = config.batch_size * config.max_context
    steps_per_epoch = max(1, len(train_data) // tokens_per_step)
    
    total_steps = steps_per_epoch * config.epochs

    rngs = nnx.Rngs(config.seed)
    model = Transformer(config, rngs=rngs)
    
    def xavier_init(model):
        for path, node in nnx.iter_graph(model):
            if isinstance(node, nnx.Linear):
                node.kernel = jax.nn.initializers.glorot_uniform()(rngs.params(), node.kernel.shape)
            elif isinstance(node, nnx.Embed):
                node.embedding = jax.nn.initializers.glorot_uniform()(rngs.params(), node.embedding.shape)

    xavier_init(model)
    optimizer = nnx.Optimizer(model, get_optax_optimizer(config.optimizer, config.lr), wrt=nnx.Param)
    report_model_summary(model, config, optimizer, summary_file_path)
    
    log_f = open(log_file_path, 'a', newline='')
    log_writer = csv.writer(log_f)
    if os.path.getsize(log_file_path) == 0:
        log_writer.writerow(['step', 'train_loss', 'val_loss', 'vram_gb', 'ms_per_step'])

    micro_batch_size = config.batch_size // config.grad_accum

    def loss_fn(model, x, y):
        logits, _ = model(x, use_cache=False, kv_caches=None, cache_index=0)
        return compute_loss(logits, y)

    @nnx.jit
    def train_step(model, optimizer, full_x, full_y):
        x_batches = full_x.reshape(config.grad_accum, micro_batch_size, -1)
        y_batches = full_y.reshape(config.grad_accum, micro_batch_size, -1)
        
        def compute_loss_for_microbatch(model, x, y):
            logits, _ = model(x, use_cache=False, kv_caches=None, cache_index=0)
            return compute_loss(logits, y)
            
        grad_fn = nnx.value_and_grad(compute_loss_for_microbatch)
        
        grad_acc = jax.tree_util.tree_map(jnp.zeros_like, nnx.state(model, nnx.Param))
        total_loss = jnp.array(0.0)
        
        for i in range(config.grad_accum):
            loss, grads = grad_fn(model, x_batches[i], y_batches[i])
            
            grad_acc = jax.tree_util.tree_map(
                lambda acc, g: acc + g / config.grad_accum, grad_acc, grads
            )
            total_loss += loss / config.grad_accum
            
        optimizer.update(model, grad_acc)
        return total_loss

    @nnx.jit
    def eval_step(model, x, y):
        return loss_fn(model, x, y)

    def estimate_loss(key):
        out = {}
        for split, d in [('train', train_data), ('val', val_data)]:
            losses = []
            for k in range(config.eval_iters):
                key, subkey = jax.random.split(key)
                x, y = get_batch(d, 1, config.max_context, subkey) 
                
                step_loss = float(eval_step(model, x, y))
                losses.append(step_loss)
                
            out[split] = sum(losses) / len(losses)
        return out, key

    key = jax.random.PRNGKey(config.seed)
    t0 = time.time()
    try:
        for step in range(total_steps):
            key, subkey = jax.random.split(key)
            x, y = get_batch(train_data, config.batch_size, config.max_context, subkey)
            train_step(model, optimizer, x, y)
            if step % 50 == 0:
                t1 = time.time()
                dt = (t1 - t0) * 1000 / 50
                t0 = t1
                vram = get_vram_usage()
                losses, key = estimate_loss(key)
                print(f"Step {step:5d}/{total_steps} | Train: {losses['train']:.4f} | Val: {losses['val']:.4f} | VRAM: {vram:.2f}GB")
                log_writer.writerow([step, float(losses['train']), float(losses['val']), round(vram, 3), round(dt, 2)])
                log_f.flush()
        print("Saving model weights...")
        final_state = nnx.state(model)
        with open(os.path.join(run_dir, "model_weights.msgpack"), "wb") as f:
            import flax.serialization
            f.write(flax.serialization.msgpack_serialize(final_state))
        print(f"Model saved to: {run_dir}")
    finally:
        log_f.close()

if __name__ == "__main__":
    main()