#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_lumi import LumiShowcaseNet  # noqa: E402
from latent_lumi.data_stream import build_meta, make_lumi_loader, move_batch  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Latent_lumi from tensor files.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--token-width", type=int, default=192)
    parser.add_argument("--control-width", type=int, default=64)
    parser.add_argument("--latent-width", type=int, default=192)
    parser.add_argument("--latent-slots", type=int, default=96)
    parser.add_argument("--query-feat-width", type=int, default=2)
    parser.add_argument("--out-width", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--save-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    loader = make_lumi_loader(
        args.data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    model = LumiShowcaseNet(
        token_width=args.token_width,
        control_width=args.control_width,
        latent_width=args.latent_width,
        latent_slots=args.latent_slots,
        query_feat_width=args.query_feat_width,
        out_width=args.out_width,
        num_heads=args.num_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        sample_count = 0
        model.train()
        for batch in loader:
            batch = move_batch(batch, device)
            pred, _ = model(
                history=batch["history"],
                meta=build_meta(batch),
                query_coords=batch["query_coords"],
                query_feats=batch.get("query_feats"),
                query_geom=batch.get("query_geom"),
                return_aux=False,
            )
            loss = F.mse_loss(pred, batch["target"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = int(batch["history"].size(0))
            epoch_loss += float(loss.item()) * batch_size
            sample_count += batch_size
            global_step += 1

        mean_loss = epoch_loss / max(sample_count, 1)
        print(f"[epoch {epoch:03d}] loss={mean_loss:.6f} samples={sample_count} steps={global_step}")

    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "args": vars(args)}, args.save_path)
        print(f"saved={args.save_path}")


if __name__ == "__main__":
    main()
