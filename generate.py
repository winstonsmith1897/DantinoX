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
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
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
    rngs = nnx.Rngs(args.seed)
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