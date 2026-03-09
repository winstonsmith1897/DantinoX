import argparse
import jax
import jax.numpy as jnp
from flax import nnx
import time
import os
import flax.serialization
from core import Transformer, Config, generate
from utils import get_tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Nel mezzo del cammin ")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    
    config_path = os.path.join(args.run_dir, "config.yaml")
    weights_path = os.path.join(args.run_dir, "model_weights.msgpack")
    
    config = Config.from_yaml(config_path)
    
    tokenizer = get_tokenizer(config.tokenizer_type, text=" ")
    
    rngs = nnx.Rngs(args.seed)
    model = Transformer(config, rngs=rngs)
    
    if os.path.exists(weights_path):
        with open(weights_path, "rb") as f:
            state_dict = flax.serialization.msgpack_restore(f.read())
        nnx.update(model, state_dict)
    
    tokens = tokenizer.encode(args.prompt)
    x = jnp.array([tokens], dtype=jnp.int32)
    
    print(f"\nRun: {args.run_dir}")
    print(f"Prompt: {args.prompt}")
    print("-" * 30)
    
    t0 = time.time()
    
    output_tokens = generate(
        model=model,
        x=x,
        max_generations=args.max_new_tokens,
        greedy=args.greedy,
        seed=args.seed
    )
    
    t1 = time.time()
    
    generated_text = tokenizer.decode(output_tokens[0].tolist())
    duration = t1 - t0
    num_tokens = len(output_tokens[0]) - len(tokens)
    tok_per_sec = num_tokens / duration if duration > 0 else 0
    
    print(generated_text)
    print("-" * 30)
    
    print(f"INFERENCE METRICS")
    print(f"Generated tokens:  {num_tokens}")
    print(f"Total time:        {duration:.4f}s")
    print(f"Throughput:        {tok_per_sec:.2f} tok/s")
    
    kv_mem = (config.num_blocks * 2 * config.max_context * config.head_size * config.n_heads * 4) / 1e6
    print(f"KV Cache VRAM:     ~{kv_mem:.2f} MB")
    
if __name__ == "__main__":
    main()