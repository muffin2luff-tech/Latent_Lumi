from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


def _norm_coords(coords: Tensor) -> Tensor:
    low = coords.amin(dim=1, keepdim=True)
    high = coords.amax(dim=1, keepdim=True)
    span = (high - low).clamp_min(1e-6)
    return ((coords - low) / span) * 2.0 - 1.0


def _fourier_coords(coords: Tensor, bands: int, max_freq: float) -> Tensor:
    if bands <= 0:
        return coords
    normed = _norm_coords(coords)
    scales = torch.linspace(1.0, max_freq / 2.0, bands, device=coords.device, dtype=coords.dtype)
    expanded = normed.unsqueeze(-1) * scales * math.pi
    packed = torch.cat([expanded.sin(), expanded.cos(), normed.unsqueeze(-1)], dim=-1)
    return packed.flatten(start_dim=-2)


class ProbeReadout(nn.Module):
    """Point-conditioned value readout over latent bank."""

    def __init__(
        self,
        latent_width: int,
        query_feat_width: int,
        out_width: int,
        query_width: int = 192,
        num_heads: int = 4,
        control_width: int = 0,
        fourier_bands: int = 4,
        fourier_max_freq: float = 10.0,
    ):
        super().__init__()
        if query_width % num_heads != 0:
            raise ValueError("query_width must be divisible by num_heads")
        if latent_width <= 0 or out_width <= 0 or query_width <= 0:
            raise ValueError("latent_width/out_width/query_width must be > 0")
        if query_feat_width < 0 or control_width < 0:
            raise ValueError("query_feat_width/control_width must be >= 0")

        self.latent_width = int(latent_width)
        self.query_feat_width = int(query_feat_width)
        self.control_width = int(control_width)
        self.fourier_bands = int(fourier_bands)
        self.fourier_max_freq = float(fourier_max_freq)

        self.query_mlp = nn.Sequential(
            nn.LazyLinear(query_width),
            nn.GELU(),
            nn.Linear(query_width, query_width),
        )
        self.cross = nn.MultiheadAttention(
            embed_dim=query_width,
            num_heads=num_heads,
            batch_first=True,
        )
        self.context_proj = nn.Linear(latent_width, query_width)
        self.out_mlp = nn.Sequential(
            nn.LayerNorm(query_width),
            nn.Linear(query_width, query_width),
            nn.GELU(),
            nn.Linear(query_width, out_width),
        )
        self.refine = nn.Sequential(
            nn.LayerNorm(query_width),
            nn.Linear(query_width, query_width * 2),
            nn.GELU(),
            nn.Linear(query_width * 2, query_width),
        )
        self.refine_gate = nn.Parameter(torch.tensor(-0.5))

    def _build_query_tokens(
        self,
        query_coords: Tensor,
        query_feats: Optional[Tensor],
        query_geom: Optional[Tensor],
        control_vec: Optional[Tensor],
    ) -> Tensor:
        if query_coords.ndim != 3:
            raise ValueError("query_coords must be [B, N, D]")
        pieces = [_fourier_coords(query_coords, self.fourier_bands, self.fourier_max_freq)]

        if query_feats is None and self.query_feat_width > 0:
            query_feats = query_coords.new_zeros(query_coords.size(0), query_coords.size(1), self.query_feat_width)
        if query_feats is not None:
            if query_feats.ndim != 3 or query_feats.shape[:2] != query_coords.shape[:2]:
                raise ValueError("query_feats must be [B, N, F] and align with query_coords")
            if query_feats.size(-1) != self.query_feat_width:
                raise ValueError(f"query_feats last dim must be {self.query_feat_width}")
            pieces.insert(0, query_feats)
        if query_geom is not None:
            if query_geom.ndim != 3 or query_geom.shape[:2] != query_coords.shape[:2]:
                raise ValueError("query_geom must be [B, N, G] and align with query_coords")
            pieces.append(query_geom)
        if control_vec is not None:
            if self.control_width == 0:
                raise ValueError("control_vec provided but readout initialized with control_width=0")
            if control_vec.ndim != 2 or control_vec.size(-1) != self.control_width:
                raise ValueError(f"control_vec must be [B, {self.control_width}]")
            pieces.append(control_vec.unsqueeze(1).expand(-1, query_coords.size(1), -1))
        return self.query_mlp(torch.cat(pieces, dim=-1))

    def forward(
        self,
        latent_bank: Tensor,
        query_coords: Tensor,
        query_feats: Optional[Tensor] = None,
        query_geom: Optional[Tensor] = None,
        control_vec: Optional[Tensor] = None,
        return_aux: bool = False,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if latent_bank.ndim != 3:
            raise ValueError("latent_bank must be [B, L, H]")
        if latent_bank.size(0) != query_coords.size(0):
            raise ValueError("batch mismatch between latent_bank and query_coords")
        if latent_bank.size(-1) != self.latent_width:
            raise ValueError(f"latent_bank last dim must be {self.latent_width}")

        query_tokens = self._build_query_tokens(query_coords, query_feats, query_geom, control_vec)
        context_tokens = self.context_proj(latent_bank)
        readout, attn = self.cross(
            query=query_tokens,
            key=context_tokens,
            value=context_tokens,
            need_weights=return_aux,
            average_attn_weights=False,
        )

        refine_gain = torch.sigmoid(self.refine_gate)
        hidden = query_tokens + readout
        hidden = hidden + refine_gain * self.refine(hidden)
        prediction = self.out_mlp(hidden)

        if not return_aux:
            return prediction, None
        aux: Dict[str, Tensor] = {
            "query_to_latent_attn": attn if attn is not None else torch.empty(0, device=prediction.device),
            "refine_gain": refine_gain.to(device=prediction.device, dtype=prediction.dtype),
            "query_norm_mean": query_tokens.norm(dim=-1).mean(dim=-1),
        }
        return prediction, aux
