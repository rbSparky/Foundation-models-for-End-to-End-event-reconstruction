from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from lgatr.layers import EquiLinear

from ..configs import LorentzParTConfig
from .classifier import ClassAttentionBlock, Classifier
from .feedforward import Feedforward
from .processor import InteractionEmbedding, ParticleProcessor


class LorentzEquivariantAdapter(nn.Module):
    """Parameter-efficient Lorentz-equivariant residual adapter."""

    def __init__(
        self,
        embed_dim: int,
        adapter_width: int = 1,
        adapter_hidden_dim: int = 48,
        adapter_rank: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        bottleneck = adapter_rank if adapter_rank and adapter_rank > 0 else adapter_hidden_dim

        self.norm = nn.LayerNorm(embed_dim)
        self.down = nn.Linear(embed_dim, bottleneck, bias=False)
        self.to_mv = nn.Linear(bottleneck, adapter_width * 16, bias=False)
        self.equilinear = EquiLinear(
            in_mv_channels=adapter_width,
            out_mv_channels=adapter_width,
            in_s_channels=None,
            out_s_channels=None,
        )
        self.from_mv = nn.Linear(adapter_width * 16, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(1e-3, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        bsz, n_particles, _ = x.shape
        h = self.norm(x)
        h = F.gelu(self.down(h))
        mv = self.to_mv(h).view(bsz, n_particles, -1, 16)
        mv, _ = self.equilinear(mv)
        h = mv.reshape(bsz, n_particles, -1)
        h = F.gelu(self.from_mv(h))
        h = self.dropout(self.up(h))
        return x + self.scale * h


class AdapterParticleAttentionBlock(nn.Module):
    """Particle attention block that supports adapter insertion points."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        expansion_factor: int = 4,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.pmha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.feedforward = Feedforward(
            embed_dim=embed_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor,
        U: Optional[Tensor] = None,
        adapter: Optional[nn.Module] = None,
        adapter_position: str = "after_ffn",
    ) -> Tensor:
        residual = x
        x = self.layernorm1(x)
        x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        x = self.layernorm2(x)
        x = self.dropout(x)
        x = x + residual

        if adapter is not None and adapter_position == "after_attention":
            x = adapter(x)

        x = self.feedforward(x)

        if adapter is not None and adapter_position == "after_ffn":
            x = adapter(x)

        return x


class LorentzParTAdaptersEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 8,
        dropout: float = 0.1,
        expansion_factor: int = 4,
        pair_embed_dims: List[int] = [64, 64, 64],
        adapter_every_n_layers: int = 2,
        adapter_width: int = 1,
        adapter_hidden_dim: int = 48,
        adapter_rank: int = 0,
        adapter_dropout: float = 0.0,
        adapter_position: str = "after_ffn",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.adapter_every_n_layers = max(int(adapter_every_n_layers), 1)
        self.adapter_position = adapter_position
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.proj = nn.Linear(16, embed_dim)
        self.interaction_embed = InteractionEmbedding(
            num_interaction_features=4,
            pair_embed_dims=pair_embed_dims + [num_heads],
        )

        self.blocks = nn.ModuleList(
            [
                AdapterParticleAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    expansion_factor=expansion_factor,
                )
                for _ in range(num_layers)
            ]
        )

        self.adapters = nn.ModuleDict()
        for i in range(num_layers):
            if i % self.adapter_every_n_layers == 0:
                self.adapters[str(i)] = LorentzEquivariantAdapter(
                    embed_dim=embed_dim,
                    adapter_width=adapter_width,
                    adapter_hidden_dim=adapter_hidden_dim,
                    adapter_rank=adapter_rank,
                    dropout=adapter_dropout,
                )

    def forward(self, x: Tensor, padding_mask: Tensor, U: Tensor) -> Tensor:
        U = self.interaction_embed(U)
        x = self.proj(x)

        for i, block in enumerate(self.blocks):
            key = str(i)
            adapter = self.adapters[key] if key in self.adapters else None

            if self.gradient_checkpointing and self.training:
                def _block_forward(
                    x_in: Tensor,
                    *,
                    _block=block,
                    _adapter=adapter,
                    _padding_mask=padding_mask,
                    _U=U,
                ) -> Tensor:
                    return _block(
                        x_in,
                        _padding_mask,
                        _U,
                        adapter=_adapter,
                        adapter_position=self.adapter_position,
                    )

                x = torch_checkpoint(_block_forward, x, use_reentrant=False)
            else:
                x = block(x, padding_mask, U, adapter=adapter, adapter_position=self.adapter_position)

        return x


class LorentzParTAdapters(nn.Module):
    """ParT backbone with lightweight Lorentz-equivariant adapters."""

    def __init__(
        self,
        config: Optional[LorentzParTConfig] = None,
    ):
        super().__init__()
        cfg = config if config is not None else LorentzParTConfig(model_variant="lorentz_adapters")

        self.max_num_particles = cfg.max_num_particles
        self.num_particle_features = cfg.num_particle_features
        self.num_classes = cfg.num_classes
        self.embed_dim = cfg.embed_dim
        self.num_heads = cfg.num_heads
        self.num_layers = cfg.num_layers
        self.num_cls_layers = cfg.num_cls_layers
        self.num_mlp_layers = cfg.num_mlp_layers
        self.hidden_dim = cfg.hidden_dim
        self.dropout = cfg.dropout
        self.expansion_factor = cfg.expansion_factor
        self.pair_embed_dims = cfg.pair_embed_dims
        self.mask = cfg.mask
        self.weights = cfg.weights
        self.inference = cfg.inference
        self.reconstruction_dim = int(cfg.reconstruction_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim), requires_grad=True)
        nn.init.normal_(self.cls_token, mean=0.0, std=1.0)

        self.processor = ParticleProcessor(to_multivector=True)
        self.encoder = LorentzParTAdaptersEncoder(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
            expansion_factor=self.expansion_factor,
            pair_embed_dims=self.pair_embed_dims,
            adapter_every_n_layers=cfg.adapter_every_n_layers,
            adapter_width=cfg.adapter_width,
            adapter_hidden_dim=cfg.adapter_hidden_dim,
            adapter_rank=cfg.adapter_rank,
            adapter_dropout=cfg.adapter_dropout,
            adapter_position=cfg.adapter_position,
            gradient_checkpointing=cfg.gradient_checkpointing,
        )

        self.fc = nn.Linear(self.max_num_particles * self.embed_dim, self.reconstruction_dim)

        self.decoder = nn.ModuleList(
            [
                ClassAttentionBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    dropout=0.0,
                    expansion_factor=self.expansion_factor,
                )
                for _ in range(self.num_cls_layers)
            ]
        )
        self.layernorm = nn.LayerNorm(self.embed_dim)
        self.classifier = Classifier(
            num_classes=self.num_classes,
            input_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_mlp_layers,
            dropout=self.dropout,
        )
        self.act = nn.Softmax(dim=1) if self.inference else nn.Identity()

        if self.weights:
            state_dict = torch.load(self.weights, map_location="cpu")
            filtered = {k[len("encoder.") :]: v for k, v in state_dict.items() if k.startswith("encoder.")}
            self.encoder.load_state_dict(filtered, strict=False)

    def forward(self, x: Tensor, mask_idx: Optional[Tensor] = None) -> Tensor:
        bsz, _, _ = x.shape
        padding_mask = (x[..., 3] == 0).float()
        if mask_idx is not None:
            batch_indices = torch.arange(x.size(0), device=x.device)
            padding_mask[batch_indices, mask_idx] = 0.0

        x, U = self.processor(x)
        x = self.encoder(x, padding_mask, U)

        if not self.mask:
            x_cls = self.cls_token.expand(bsz, -1, -1)
            for layer in self.decoder:
                x_cls = layer(x, x_cls, padding_mask)
            x_cls = self.layernorm(x_cls).squeeze(1)
            x_cls = self.classifier(x_cls)
            return self.act(x_cls)

        x = x.reshape(bsz, -1)
        return self.fc(x)
