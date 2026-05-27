from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn

from .attention_blender import SpatialTemporalBridge, TokenBlendGate
from .query_readout import ProbeReadout
from .state_codec import SignalAssembler


class LumiBackbone(nn.Module):
    """Small latent core for showcase usage."""

    def __init__(
        self,
        token_width: int = 192,
        control_width: int = 64,
        latent_width: int = 192,
        latent_slots: int = 96,
        num_heads: int = 4,
    ):
        super().__init__()
        if latent_width % num_heads != 0:
            raise ValueError("latent_width must be divisible by num_heads")

        self.token_norm = nn.LayerNorm(token_width)
        self.token_proj = nn.Linear(token_width, latent_width)
        self.seed_latents = nn.Parameter(torch.randn(latent_slots, latent_width) * 0.02)

        self.token_to_latent = nn.MultiheadAttention(
            embed_dim=latent_width,
            num_heads=num_heads,
            batch_first=True,
        )
        self.temporal_delta = nn.Sequential(
            nn.LayerNorm(latent_width),
            nn.Linear(latent_width, latent_width * 2),
            nn.SiLU(),
            nn.Linear(latent_width * 2, latent_width),
        )
        self.spatial_block = nn.MultiheadAttention(
            embed_dim=latent_width,
            num_heads=num_heads,
            batch_first=True,
        )
        self.spatial_norm = nn.LayerNorm(latent_width)
        self.align = SpatialTemporalBridge(
            spatial_width=latent_width,
            temporal_width=latent_width,
            temporal_slots=latent_slots,
            num_heads=num_heads,
        )
        self.gate = TokenBlendGate(
            latent_width=latent_width,
            control_width=control_width,
            hidden_width=latent_width,
        )

    def forward(
        self,
        token_seq: Tensor,
        control_vec: Optional[Tensor] = None,
        return_aux: bool = False,
    ) -> Dict[str, Tensor]:
        if token_seq.ndim != 3:
            raise ValueError("token_seq must be [B, N, C]")
        bsz = token_seq.size(0)

        memory = self.token_proj(self.token_norm(token_seq))
        query = self.seed_latents.unsqueeze(0).expand(bsz, -1, -1)
        latent_base, _ = self.token_to_latent(query=query, key=memory, value=memory, need_weights=False)

        delta_t = self.temporal_delta(latent_base)
        spatial_ctx, _ = self.spatial_block(
            query=latent_base,
            key=latent_base,
            value=latent_base,
            need_weights=False,
        )
        delta_s = self.spatial_norm(spatial_ctx - latent_base)
        delta_s_aligned = self.align(delta_s)
        gate, gate_aux = self.gate(
            latent_base=latent_base,
            temporal_delta=delta_t,
            spatial_delta=delta_s_aligned,
            control_vec=control_vec,
            with_stats=return_aux,
        )
        latent_next = latent_base + gate * delta_t + (1.0 - gate) * delta_s_aligned

        out: Dict[str, Tensor] = {
            "latent_bank": latent_next,
            "gate": gate,
        }
        if return_aux and gate_aux is not None:
            out["gate_mean"] = gate_aux["gate_mean"]
            out["gate_entropy"] = gate_aux["gate_entropy"]
        return out


class LumiShowcaseNet(nn.Module):
    """End-to-end synthetic-ready model for reviewer-facing demo."""

    def __init__(
        self,
        token_width: int = 192,
        control_width: int = 64,
        latent_width: int = 192,
        latent_slots: int = 96,
        query_feat_width: int = 2,
        out_width: int = 1,
        num_heads: int = 4,
    ):
        super().__init__()
        self.assembler = SignalAssembler(
            token_width=token_width,
            control_width=control_width,
            hidden_width=128,
            include_coords=True,
            include_geom=True,
            include_boundary=True,
            include_source=True,
            include_grad=True,
            light_state_bins=4,
            history_mode="last_mean_flat",
        )
        self.backbone = LumiBackbone(
            token_width=token_width,
            control_width=control_width,
            latent_width=latent_width,
            latent_slots=latent_slots,
            num_heads=num_heads,
        )
        self.readout = ProbeReadout(
            latent_width=latent_width,
            query_feat_width=query_feat_width,
            out_width=out_width,
            query_width=latent_width,
            num_heads=num_heads,
            control_width=control_width,
            fourier_bands=4,
            fourier_max_freq=10.0,
        )

    def forward(
        self,
        history: Tensor,
        meta: Mapping[str, Tensor],
        query_coords: Tensor,
        query_feats: Optional[Tensor] = None,
        query_geom: Optional[Tensor] = None,
        return_aux: bool = False,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        encoded = self.assembler(history, meta)
        core_out = self.backbone(
            token_seq=encoded["token_seq"],
            control_vec=encoded.get("control_vec"),
            return_aux=return_aux,
        )
        prediction, read_aux = self.readout(
            latent_bank=core_out["latent_bank"],
            query_coords=query_coords,
            query_feats=query_feats,
            query_geom=query_geom,
            control_vec=encoded.get("control_vec"),
            return_aux=return_aux,
        )

        if not return_aux:
            return prediction, None
        aux: Dict[str, Tensor] = {
            "gate": core_out["gate"],
        }
        if "gate_mean" in core_out:
            aux["gate_mean"] = core_out["gate_mean"]
        if "gate_entropy" in core_out:
            aux["gate_entropy"] = core_out["gate_entropy"]
        if read_aux is not None:
            for key, value in read_aux.items():
                aux[f"readout_{key}"] = value
        return prediction, aux
