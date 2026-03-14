import argparse
import jax
import jax.numpy as jnp
from flax import nnx
import time
import os
import msgpack
import yaml
import flax.serialization
from core import Transformer, Config, generate
from utils import get_tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Nel mezzo del cammin ")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--greedy", type=str, choices=['true', 'false'], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--use_cache", type=str, choices=['true', 'false'], default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    
    config_path = os.path.join(args.run_dir, "config.yaml")
    weights_path = os.path.join(args.run_dir, "model_weights.msgpack")
    
    with open(config_path, 'r') as f:
        raw_cfg = yaml.safe_load(f)
    
    if any(isinstance(v, dict) for v in raw_cfg.values()):
        flat_cfg = {}
        for section in raw_cfg.values():
            if isinstance(section, dict): flat_cfg.update(section)
    else:
        flat_cfg = raw_cfg
    
    config = Config(**{k: v for k, v in flat_cfg.items() if k in Config.__dataclass_fields__})
    
    if config.dataset_source == "huggingface":
        from datasets import load_dataset
        raw_dataset = load_dataset(config.dataset_name, split='train')
        text = " ".join(raw_dataset['text'])
    else:
        with open(config.dataset_name, "r", encoding="utf-8") as f:
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
    
    gen_seed = flat_cfg.get('seed', 42) if args.seed is None else args.seed
    
    rngs = nnx.Rngs(gen_seed)
    model = Transformer(config, rngs=rngs)
    
    if os.path.exists(weights_path):
        with open(weights_path, "rb") as f:
            from flax.serialization import _msgpack_ext_unpack
            raw_data = f.read()
            state_dict = msgpack.unpackb(
                raw_data, 
                ext_hook=_msgpack_ext_unpack, 
                strict_map_key=False
            )
        nnx.update(model, state_dict)

    tokens = tokenizer.encode(args.prompt)
    x = jnp.array([tokens], dtype=jnp.int32)
    
    print(f"\nRun: {args.run_dir}")
    print(f"Prompt: {args.prompt}")
    print("-" * 30)

    gen_max_new_tokens = flat_cfg.get('max_generations', 150) if args.max_new_tokens is None else args.max_new_tokens
    gen_greedy = flat_cfg.get('greedy', False) if args.greedy is None else (args.greedy == 'true')
    gen_top_k = flat_cfg.get('top_k', None) if args.top_k is None else args.top_k
    gen_top_p = flat_cfg.get('top_p', None) if args.top_p is None else args.top_p
    gen_temperature = flat_cfg.get('temperature', 1.0) if args.temperature is None else args.temperature
    gen_use_cache = flat_cfg.get('use_cache', True) if args.use_cache is None else (args.use_cache == 'true')

    _ = generate(
        model=model,
        x=x,
        max_generations=1,
        greedy=gen_greedy,
        seed=gen_seed,
        use_cache=gen_use_cache,
        top_k=gen_top_k,
        top_p=gen_top_p,
        temperature=gen_temperature
    )
    x.block_until_ready()

    t0 = time.time()
    
    output_tokens = generate(
        model=model,
        x=x,
        max_generations=gen_max_new_tokens,
        greedy=gen_greedy,
        seed=gen_seed,
        use_cache=gen_use_cache,
        top_k=gen_top_k,
        top_p=gen_top_p,
        temperature=gen_temperature
    )
    
    output_tokens.block_until_ready()
    
    t1 = time.time()
    
    generated_text = tokenizer.decode(output_tokens[0].tolist())
    if config.tokenizer_type == "bpe":
        generated_text = generated_text.replace(" ", "")
        generated_text = generated_text.replace("Ġ", " ").replace("âĢĻ", "’").replace("Ã¹", "ù").replace("Ã¬", "ì").replace("Ã©", "é").replace("Ã¨", "è").replace("Ã²", "ò").replace("Ã", "à")
    
    duration = t1 - t0
    num_tokens = len(output_tokens[0]) - len(tokens)
    tok_per_sec = num_tokens / duration if duration > 0 else 0
    
    print(generated_text)
    print("-" * 30)
    print(f"INFERENCE METRICS")
    print(f"Generated tokens:  {num_tokens}")
    print(f"Total time:        {duration:.4f}s")
    print(f"Throughput:        {tok_per_sec:.2f} tok/s")
    
if __name__ == "__main__":
    main()