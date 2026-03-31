from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Tuple


@dataclass
class BaseModelConfig:
    model_variant: str = "part"  # part | lorentz_2eq | lorentz_adapters
    num_classes: int = 10
    embed_dim: int = 128
    num_heads: int = 8
    num_layers: int = 8
    num_cls_layers: int = 2
    num_mlp_layers: int = 0
    hidden_dim: int = 256
    dropout: float = 0.1
    max_num_particles: int = 128
    num_particle_features: int = 4
    expansion_factor: int = 4
    mask: bool = False
    weights: Optional[str] = None
    inference: bool = False


@dataclass
class ParticleTransformerConfig(BaseModelConfig):
    pair_embed_dims: List[int] = field(default_factory=lambda: [64, 64, 64])

    @classmethod
    def from_dict(cls, d: Dict):
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class LGATrConfig(BaseModelConfig):
    hidden_mv_channels: int = 8
    in_s_channels: int = None
    out_s_channels: int = None
    hidden_s_channels: int = 16
    attention: Optional[Dict] = None
    mlp: Optional[Dict] = None
    reinsert_mv_channels: Optional[Tuple[int]] = None
    reinsert_s_channels: Optional[Tuple[int]] = None

    @classmethod
    def from_dict(cls, d: Dict):
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class LorentzParTConfig(BaseModelConfig):
    hidden_mv_channels: int = 8
    in_s_channels: int = None
    out_s_channels: int = None
    hidden_s_channels: int = 16
    attention: Optional[Dict] = None
    mlp: Optional[Dict] = None
    reinsert_mv_channels: Optional[Tuple[int]] = None
    reinsert_s_channels: Optional[Tuple[int]] = None
    pair_embed_dims: List[int] = field(default_factory=lambda: [64, 64, 64])

    # Adapter controls (used when model_variant == 'lorentz_adapters')
    adapter_every_n_layers: int = 2
    adapter_width: int = 1
    adapter_hidden_dim: int = 48
    adapter_rank: int = 0
    adapter_dropout: float = 0.0
    adapter_position: str = "after_ffn"  # after_attention | after_ffn
    reconstruction_dim: int = 4
    gradient_checkpointing: bool = False

    @classmethod
    def from_dict(cls, d: Dict):
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})
