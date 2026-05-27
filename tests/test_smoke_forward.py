from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_lumi import LumiShowcaseNet  # noqa: E402


def test_forward_shape_smoke() -> None:
    model = LumiShowcaseNet(
        token_width=64,
        control_width=32,
        latent_width=64,
        latent_slots=24,
        query_feat_width=2,
        out_width=1,
        num_heads=4,
    )
    bsz, steps, points = 2, 6, 16
    history = torch.rand(bsz, steps, points, 1)
    coords = torch.rand(bsz, points, 2)
    meta = {
        "coords": coords,
        "point_geom": torch.rand(bsz, points, 2),
        "point_boundary": torch.rand(bsz, points, 1),
        "point_source": torch.rand(bsz, points, 1),
        "point_grad": torch.rand(bsz, points, 1),
        "shade": torch.rand(bsz, 1),
        "blind": torch.rand(bsz, 1) * 90.0,
        "light": torch.randint(0, 4, (bsz, 1)).float(),
        "extra_control": torch.rand(bsz, 2),
    }
    query_coords = torch.rand(bsz, points, 2)
    query_feats = torch.rand(bsz, points, 2)
    query_geom = torch.rand(bsz, points, 2)

    pred, aux = model(
        history=history,
        meta=meta,
        query_coords=query_coords,
        query_feats=query_feats,
        query_geom=query_geom,
        return_aux=True,
    )

    assert pred.shape == (bsz, points, 1)
    assert aux is not None
    assert "gate_mean" in aux
