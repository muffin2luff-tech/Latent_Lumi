from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


class SpatialTemporalBridge(nn.Module):
    """Maps a spatial update branch into temporal latent coordinates."""

    def __init__(
        self,
        spatial_width: int,
        temporal_width: int,
        temporal_slots: int,
        num_heads: int = 4,
    ):
        super().__init__()
        if temporal_width % num_heads != 0:
            raise ValueError("temporal_width must be divisible by num_heads")
        self.spatial_width = int(spatial_width)
        self.temporal_width = int(temporal_width)
        self.temporal_slots = int(temporal_slots)

        self.spatial_proj = nn.Linear(spatial_width, temporal_width)
        self.latent_queries = nn.Parameter(torch.randn(temporal_slots, temporal_width) * 0.02)
        self.cross_map = nn.MultiheadAttention(
            embed_dim=temporal_width,
            num_heads=num_heads,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(temporal_width)

    def forward(self, spatial_delta: Tensor) -> Tensor:
        if spatial_delta.ndim != 3:
            raise ValueError(f"spatial_delta must be [B, S, D], got {tuple(spatial_delta.shape)}")
        if spatial_delta.size(-1) != self.spatial_width:
            raise ValueError(
                f"spatial width mismatch: expected {self.spatial_width}, got {spatial_delta.size(-1)}"
            )
        bsz = spatial_delta.size(0)
        lifted = self.spatial_proj(spatial_delta)
        if lifted.size(1) == self.temporal_slots:
            return self.out_norm(lifted)

        query = self.latent_queries.unsqueeze(0).expand(bsz, -1, -1)
        aligned, _ = self.cross_map(query=query, key=lifted, value=lifted, need_weights=False)
        return self.out_norm(aligned)


class TokenBlendGate(nn.Module):
    """Token-wise gate to blend temporal and spatial corrections."""

    def __init__(
        self,
        latent_width: int,
        control_width: int = 0,
        hidden_width: int = 128,
    ):
        super().__init__()
        self.latent_width = int(latent_width)
        self.control_width = int(control_width)

        if self.control_width > 0:
            self.control_proj = nn.Sequential(
                nn.LayerNorm(control_width),
                nn.Linear(control_width, latent_width),
                nn.SiLU(),
                nn.Linear(latent_width, latent_width),
            )
            gate_input_width = 4 * latent_width
        else:
            self.control_proj = None
            gate_input_width = 3 * latent_width

        self.gate_body = nn.Sequential(
            nn.LayerNorm(gate_input_width),
            nn.Linear(gate_input_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 1),
        )

        final_layer = self.gate_body[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(
        self,
        latent_base: Tensor,
        temporal_delta: Tensor,
        spatial_delta: Tensor,
        control_vec: Optional[Tensor] = None,
        with_stats: bool = False,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if latent_base.ndim != 3 or temporal_delta.ndim != 3 or spatial_delta.ndim != 3:
            raise ValueError("latent_base/temporal_delta/spatial_delta must all be [B, L, H]")
        if latent_base.shape != temporal_delta.shape or latent_base.shape != spatial_delta.shape:
            raise ValueError("all latent tensors must share shape")
        if latent_base.size(-1) != self.latent_width:
            raise ValueError("latent width mismatch")

        inputs = [latent_base, temporal_delta, spatial_delta]
        if self.control_proj is not None:
            if control_vec is None:
                raise ValueError("control_vec is required when control_width > 0")
            if control_vec.ndim != 2 or control_vec.size(-1) != self.control_width:
                raise ValueError(f"control_vec must be [B, {self.control_width}]")
            control_expand = self.control_proj(control_vec).unsqueeze(1).expand(-1, latent_base.size(1), -1)
            inputs.append(control_expand)

        gate_logits = self.gate_body(torch.cat(inputs, dim=-1))
        gate = torch.sigmoid(gate_logits)
        if not with_stats:
            return gate, None

        clipped = gate.clamp(1e-6, 1.0 - 1e-6)
        entropy = -clipped * clipped.log() - (1.0 - clipped) * (1.0 - clipped).log()
        stats: Dict[str, Tensor] = {
            "gate_mean": gate.mean(dim=(1, 2)),
            "gate_entropy": entropy.mean(dim=(1, 2)),
            "gate_logits": gate_logits,
        }
        return gate, stats
