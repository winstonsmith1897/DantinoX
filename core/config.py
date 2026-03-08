from dataclasses import dataclass, asdict
import yaml


@dataclass
class Config:
    dim: int = 128
    n_heads: int = 16
    head_size: int = 8
    num_blocks: int = 4
    vocab_size: int = 200
    max_context: int = 110
    use_moe: bool = True
    n_experts: int = 4
    top_k_mlp: int = 2
    expansion: int = 4
    use_rotary_pos: bool = True
    sliding_window: bool = True
    context_window: int = 4
    no_sink: bool = True
    kv_heads: int = None

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
            flat_cfg.update(section)
            
        return cls(**flat_cfg)

    def save_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f)
