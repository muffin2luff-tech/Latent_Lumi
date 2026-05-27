#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_lumi import LumiShowcaseNet  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_fake_batch(
    batch_size: int,
    hist_steps: int,
    points: int,
    channels: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    history = torch.rand(batch_size, hist_steps, points, channels, device=device)
    coords = torch.rand(batch_size, points, 2, device=device)
    point_geom = torch.rand(batch_size, points, 2, device=device)
    point_boundary = torch.rand(batch_size, points, 1, device=device)
    point_source = torch.rand(batch_size, points, 1, device=device)
    point_grad = torch.rand(batch_size, points, 1, device=device)

    shade = torch.rand(batch_size, 1, device=device)
    blind = torch.rand(batch_size, 1, device=device) * 90.0
    light = torch.randint(low=0, high=4, size=(batch_size, 1), device=device).float()
    extra_control = torch.rand(batch_size, 2, device=device)

    query_coords = coords + 0.05 * torch.randn_like(coords)
    query_feats = torch.cat([history[:, -1], history.mean(dim=1)], dim=-1)
    query_geom = point_geom

    last_signal = history[:, -1, :, :1]
    trend_signal = history[:, :, :, :1].mean(dim=1)
    coord_term = 0.4 * query_coords[..., :1] + 0.2 * query_coords[..., 1:2]
    control_term = (
        0.1 * shade.unsqueeze(1) - 0.002 * blind.unsqueeze(1) + 0.02 * light.unsqueeze(1)
    )
    target = 0.7 * last_signal + 0.3 * trend_signal + coord_term + control_term

    meta = {
        "coords": coords,
        "point_geom": point_geom,
        "point_boundary": point_boundary,
        "point_source": point_source,
        "point_grad": point_grad,
        "shade": shade,
        "blind": blind,
        "light": light,
        "extra_control": extra_control,
    }
    return history, meta, query_coords, query_feats, query_geom, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Latent_lumi synthetic smoke training.")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--points", type=int, default=48)
    parser.add_argument("--hist-steps", type=int, default=12)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def pick_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    set_seed(args.seed)

    model = LumiShowcaseNet(
        token_width=192,
        control_width=64,
        latent_width=192,
        latent_slots=96,
        query_feat_width=2,
        out_width=1,
        num_heads=4,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_loss = None
    final_loss = None
    for step in range(1, args.steps + 1):
        history, meta, q_coords, q_feats, q_geom, target = make_fake_batch(
            batch_size=args.batch_size,
            hist_steps=args.hist_steps,
            points=args.points,
            channels=args.channels,
            device=device,
        )
        pred, _ = model(
            history=history,
            meta=meta,
            query_coords=q_coords,
            query_feats=q_feats,
            query_geom=q_geom,
            return_aux=False,
        )
        loss = F.mse_loss(pred, target)
        if start_loss is None:
            start_loss = float(loss.item())

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        final_loss = float(loss.item())

        if step == 1 or step == args.steps or step % 10 == 0:
            print(f"[step {step:03d}] loss={loss.item():.6f}")

    eval_history, eval_meta, eval_q_coords, eval_q_feats, eval_q_geom, _ = make_fake_batch(
        batch_size=args.batch_size,
        hist_steps=args.hist_steps,
        points=args.points,
        channels=args.channels,
        device=device,
    )
    with torch.no_grad():
        eval_pred, aux = model(
            history=eval_history,
            meta=eval_meta,
            query_coords=eval_q_coords,
            query_feats=eval_q_feats,
            query_geom=eval_q_geom,
            return_aux=True,
        )

    print(f"device={device.type}")
    print(f"prediction_shape={tuple(eval_pred.shape)}")
    if aux is not None and "gate_mean" in aux:
        print(f"gate_mean_batch={aux['gate_mean'].detach().cpu().tolist()}")
    if start_loss is not None and final_loss is not None:
        print(f"start_loss={start_loss:.6f} final_loss={final_loss:.6f}")


if __name__ == "__main__":
    main()
