from dataclasses import dataclass, asdict
import yaml

@dataclass
class Config:
    # Model Architecture
    dim: int = 512
    n_heads: int = 16
    head_size: int = 8
    num_blocks: int = 8
    vocab_size: int = 200
    max_context: int = 512
    kv_heads: int = None
    
    # MoE
    use_moe: bool = True
    n_experts: int = 4
    top_k_mlp: int = 2
    expansion: int = 4
    
    # Attention & Positional Features
    use_rotary_pos: bool = True
    trainable_pos: bool = False  
    absolute_pos: bool = False    
    sliding_window: bool = False
    context_window: int = 4
    no_sink: bool = True
    
    # Tokenizer
    tokenizer_type: str = "char"
    tokenizer_path: str = None  
    
    # Dataset Configuration
    dataset_source: str = "local"
    dataset_name: str = ""
    streaming: bool = False
    
    # Training & Optimization
    lr: float = 0.0005
    batch_size: int = 32
    grad_accum: int = 4
    steps: int = 5000
    seed: int = 42
    optimizer: str = "adamw"
    
    # Logging & Metrics
    eval_iters: int = 20
    log_file: str = "training_log.csv"
    summary_file: str = "model_summary.json"

    def __post_init__(self):
        if self.kv_heads is None:
            self.kv_heads = self.n_heads // 4
        assert self.dim == self.n_heads * self.head_size
        assert self.n_heads % self.kv_heads == 0

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            raw_cfg = yaml.safe_load(f)
        
        flat_cfg = {}
        for section in raw_cfg.values():
            if isinstance(section, dict):
                flat_cfg.update(section)
                
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered_cfg = {k: v for k, v in flat_cfg.items() if k in valid_keys}
            
        return cls(**filtered_cfg)

    def save_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f)