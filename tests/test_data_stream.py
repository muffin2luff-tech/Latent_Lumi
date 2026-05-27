from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_lumi.data_stream import build_meta, make_lumi_loader  # noqa: E402


def test_lumi_loader_reads_npz_directory(tmp_path: Path) -> None:
    for idx in range(3):
        np.savez(
            tmp_path / f"sample_{idx:03d}.npz",
            history=np.random.rand(6, 10, 1).astype("float32"),
            coords=np.random.rand(10, 2).astype("float32"),
            point_geom=np.random.rand(10, 2).astype("float32"),
            point_boundary=np.random.rand(10, 1).astype("float32"),
            point_source=np.random.rand(10, 1).astype("float32"),
            point_grad=np.random.rand(10, 1).astype("float32"),
            shade=np.random.rand(1).astype("float32"),
            blind=(np.random.rand(1) * 90.0).astype("float32"),
            light=np.random.randint(0, 4, size=(1,)).astype("float32"),
            extra_control=np.random.rand(2).astype("float32"),
            query_coords=np.random.rand(10, 2).astype("float32"),
            query_feats=np.random.rand(10, 2).astype("float32"),
            query_geom=np.random.rand(10, 2).astype("float32"),
            target=np.random.rand(10, 1).astype("float32"),
        )

    loader = make_lumi_loader(tmp_path, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    meta = build_meta(batch)

    assert batch["history"].shape == (2, 6, 10, 1)
    assert batch["target"].shape == (2, 10, 1)
    assert meta["coords"].shape == (2, 10, 2)
    assert meta["shade"].shape == (2, 1)
