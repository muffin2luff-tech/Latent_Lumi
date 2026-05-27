from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
from torch import Tensor, nn


def _as_plain_dict(meta: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if meta is None:
        return {}
    return dict(meta)


def _ensure_batch_feature(x: Optional[Tensor], *, label: str) -> Optional[Tensor]:
    if x is None:
        return None
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    if x.ndim != 2:
        raise ValueError(f"{label} must be [B] or [B, D], got {tuple(x.shape)}")
    return x


def _ensure_point_feature(
    x: Optional[Tensor],
    *,
    label: str,
    batch_size: int,
    point_count: int,
) -> Optional[Tensor]:
    if x is None:
        return None
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    if x.ndim != 3:
        raise ValueError(f"{label} must be [B, N] or [B, N, D], got {tuple(x.shape)}")
    if x.size(0) != batch_size or x.size(1) != point_count:
        raise ValueError(
            f"{label} must share [B, N]=[{batch_size}, {point_count}], got {tuple(x.shape[:2])}"
        )
    return x


class ConditionCodec(nn.Module):
    """Encodes control variables into a dense vector."""

    def __init__(
        self,
        vector_width: int,
        hidden_width: int = 128,
        prepacked_width: int = 0,
        angle_is_degree: bool = True,
        use_angle_trig: bool = True,
        light_state_bins: int = 0,
    ):
        super().__init__()
        if vector_width <= 0:
            raise ValueError("vector_width must be > 0")
        if hidden_width <= 0:
            raise ValueError("hidden_width must be > 0")
        if prepacked_width < 0:
            raise ValueError("prepacked_width must be >= 0")
        if light_state_bins < 0:
            raise ValueError("light_state_bins must be >= 0")

        self.vector_width = int(vector_width)
        self.prepacked_width = int(prepacked_width)
        self.angle_is_degree = bool(angle_is_degree)
        self.use_angle_trig = bool(use_angle_trig)
        self.light_state_bins = int(light_state_bins)

        self.body = nn.Sequential(
            nn.LazyLinear(hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, vector_width),
        )
        self.prepacked_proj = nn.Linear(prepacked_width, vector_width) if prepacked_width > 0 else None

    def _expand_angle(self, blind_angle: Tensor) -> Tensor:
        if not self.use_angle_trig:
            return blind_angle
        angle = blind_angle
        if self.angle_is_degree:
            angle = torch.deg2rad(angle)
        return torch.cat([blind_angle, angle.sin(), angle.cos()], dim=-1)

    def _expand_light(self, light_state: Tensor) -> Tensor:
        if self.light_state_bins <= 1:
            return light_state
        if light_state.size(-1) != 1:
            return light_state
        light_ids = light_state.squeeze(-1)
        if torch.is_floating_point(light_ids):
            light_ids = light_ids.round().long()
        else:
            light_ids = light_ids.long()
        light_ids = light_ids.clamp_min(0).clamp_max(self.light_state_bins - 1)
        return torch.nn.functional.one_hot(light_ids, num_classes=self.light_state_bins).to(light_state.dtype)

    def forward(
        self,
        *,
        prepacked: Optional[Tensor] = None,
        shade: Optional[Tensor] = None,
        blind: Optional[Tensor] = None,
        light: Optional[Tensor] = None,
        extra: Optional[Tensor] = None,
    ) -> Tensor:
        if prepacked is not None:
            prepacked = _ensure_batch_feature(prepacked, label="prepacked")
            if self.prepacked_proj is not None:
                if prepacked.size(-1) != self.prepacked_width:
                    raise ValueError(
                        f"prepacked last dim ({prepacked.size(-1)}) != prepacked_width ({self.prepacked_width})"
                    )
                return self.prepacked_proj(prepacked)
            return self.body(prepacked)

        fragments = []
        shade = _ensure_batch_feature(shade, label="shade")
        blind = _ensure_batch_feature(blind, label="blind")
        light = _ensure_batch_feature(light, label="light")
        extra = _ensure_batch_feature(extra, label="extra")

        if shade is not None:
            fragments.append(shade)
        if blind is not None:
            fragments.append(self._expand_angle(blind))
        if light is not None:
            fragments.append(self._expand_light(light))
        if extra is not None:
            fragments.append(extra)
        if len(fragments) == 0:
            raise ValueError("ConditionCodec needs prepacked or at least one of shade/blind/light/extra")
        return self.body(torch.cat(fragments, dim=-1))


class SignalAssembler(nn.Module):
    """Converts history + metadata into token sequence for latent processing."""

    def __init__(
        self,
        token_width: int,
        control_width: int,
        hidden_width: int = 128,
        include_coords: bool = True,
        include_geom: bool = True,
        include_boundary: bool = True,
        include_source: bool = True,
        include_grad: bool = True,
        light_state_bins: int = 0,
        history_mode: str = "last_mean_flat",
    ):
        super().__init__()
        if token_width <= 0:
            raise ValueError("token_width must be > 0")
        if control_width <= 0:
            raise ValueError("control_width must be > 0")
        if history_mode not in {"last_only", "last_mean_flat"}:
            raise ValueError("history_mode must be one of {'last_only', 'last_mean_flat'}")

        self.token_width = int(token_width)
        self.control_width = int(control_width)
        self.include_coords = bool(include_coords)
        self.include_geom = bool(include_geom)
        self.include_boundary = bool(include_boundary)
        self.include_source = bool(include_source)
        self.include_grad = bool(include_grad)
        self.history_mode = str(history_mode)

        self.control_codec = ConditionCodec(
            vector_width=control_width,
            hidden_width=hidden_width,
            light_state_bins=light_state_bins,
        )
        self.token_lift = nn.Sequential(
            nn.LazyLinear(hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, token_width),
        )

    def _collapse_history(self, history: Tensor) -> Tensor:
        if history.ndim == 3:
            return history
        if history.ndim != 4:
            raise ValueError("history must be [B, N, C] or [B, T, N, C]")
        bsz, steps, points, chans = history.shape
        latest = history[:, -1]
        if self.history_mode == "last_only":
            return latest
        mean_hist = history.mean(dim=1)
        flat_hist = history.permute(0, 2, 1, 3).reshape(bsz, points, steps * chans)
        return torch.cat([latest, mean_hist, flat_hist], dim=-1)

    def _build_control(self, meta: Mapping[str, Any]) -> Optional[Tensor]:
        if "control_vec" in meta and meta["control_vec"] is not None:
            control = meta["control_vec"]
            if not torch.is_tensor(control):
                raise ValueError("meta['control_vec'] must be Tensor when provided")
            if control.ndim != 2 or control.size(-1) != self.control_width:
                raise ValueError(f"control_vec must be [B, {self.control_width}]")
            return control

        keys = {"prepacked_control", "shade", "blind", "light", "extra_control"}
        if not any(k in meta and meta[k] is not None for k in keys):
            return None
        return self.control_codec(
            prepacked=meta.get("prepacked_control"),
            shade=meta.get("shade"),
            blind=meta.get("blind"),
            light=meta.get("light"),
            extra=meta.get("extra_control"),
        )

    def forward(self, history: Tensor, meta: Optional[Mapping[str, Any]] = None) -> Dict[str, Tensor]:
        pack = _as_plain_dict(meta)
        base_feature = self._collapse_history(history)
        bsz, points, _ = base_feature.shape

        coords = pack.get("coords")
        if coords is None:
            raise ValueError("SignalAssembler needs meta['coords']")
        if not torch.is_tensor(coords) or coords.ndim != 3:
            raise ValueError("coords must be Tensor with shape [B, N, D]")
        if coords.size(0) != bsz or coords.size(1) != points:
            raise ValueError("coords must match history [B, N]")

        point_geom = _ensure_point_feature(pack.get("point_geom"), label="point_geom", batch_size=bsz, point_count=points)
        point_boundary = _ensure_point_feature(
            pack.get("point_boundary"),
            label="point_boundary",
            batch_size=bsz,
            point_count=points,
        )
        point_source = _ensure_point_feature(pack.get("point_source"), label="point_source", batch_size=bsz, point_count=points)
        point_grad = _ensure_point_feature(pack.get("point_grad"), label="point_grad", batch_size=bsz, point_count=points)
        control_vec = self._build_control(pack)

        pieces = [base_feature]
        if self.include_coords:
            pieces.append(coords)
        if self.include_geom and point_geom is not None:
            pieces.append(point_geom)
        if self.include_boundary and point_boundary is not None:
            pieces.append(point_boundary)
        if self.include_source and point_source is not None:
            pieces.append(point_source)
        if self.include_grad and point_grad is not None:
            pieces.append(point_grad)
        if control_vec is not None:
            pieces.append(control_vec.unsqueeze(1).expand(-1, points, -1))

        token_seq = self.token_lift(torch.cat(pieces, dim=-1))
        output: Dict[str, Tensor] = {
            "token_seq": token_seq,
            "token_coords": coords,
        }
        if control_vec is not None:
            output["control_vec"] = control_vec
        if point_geom is not None:
            output["point_geom"] = point_geom
        return output
