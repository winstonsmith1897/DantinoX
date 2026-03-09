import os
import yaml
import numpy as np
from collections import Counter
from datasets import load_dataset
from core import Config

def load_config(run_dir):
    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, 'r') as f:
        raw_cfg = yaml.safe_load(f)
    
    flat_cfg = {}
    if any(isinstance(v, dict) for v in raw_cfg.values()):
        for section in raw_cfg.values():
            if isinstance(section, dict): 
                flat_cfg.update(section)
    else:
        flat_cfg = raw_cfg
            
    return Config(**{k: v for k, v in flat_cfg.items() if k in Config.__dataclass_fields__})

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.run_dir)

    if config.dataset_source == "huggingface":
        raw_dataset = load_dataset(config.dataset_name, split='train')
        text_column = 'text' if 'text' in raw_dataset.column_names else raw_dataset.column_names[0]
        text = "\n".join([str(item) for item in raw_dataset[text_column]])
    else:
        with open(config.dataset_name, "r", encoding="utf-8") as f:
            text = f.read()

    chars = len(text)
    words = text.split()
    num_words = len(words)
    lines = text.split('\n')
    num_lines = len(lines)
    
    print("\n" + "="*40)
    print(" 📊 RADIOGRAFIA DEL DATASET")
    print("="*40)
    print(f"Dataset: {config.dataset_name}")
    print(f"Caratteri totali: {chars:,}")
    print(f"Parole totali: {num_words:,}")
    print(f"Righe totali: {num_lines:,}")
    
    unique_chars = set(text)
    print(f"\nCaratteri unici: {len(unique_chars)}")
    
    counter = Counter(text)
    print("\nI 20 caratteri MENO frequenti:")
    for c, count in counter.most_common()[-20:]:
        print(f"  {repr(c):<5} : {count}")

    line_lengths = [len(l) for l in lines if len(l.strip()) > 0]
    if line_lengths:
        print("\n📏 STATISTICHE SUI VERSI (Lunghezza in caratteri):")
        print(f"Media: {np.mean(line_lengths):.1f}")
        print(f"Mediana: {np.median(line_lengths):.1f}")
        print(f"Verso più lungo: {np.max(line_lengths)}")
        print(f"Verso più corto: {np.min(line_lengths)}")
        
        anomalies = [l for l in lines if len(l) > 100]
        if anomalies:
            print(f"\n⚠️ Trovate {len(anomalies)} righe più lunghe di 100 caratteri!")
            print("Esempio riga anomala:", repr(anomalies[0][:100] + "..."))

    print("="*40 + "\n")

if __name__ == "__main__":
    main()