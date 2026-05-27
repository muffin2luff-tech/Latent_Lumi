from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


REQUIRED_KEYS = ("history", "coords", "query_coords", "target")
OPTIONAL_KEYS = (
    "point_geom",
    "point_boundary",
    "point_source",
    "point_grad",
    "shade",
    "blind",
    "light",
    "extra_control",
    "control_vec",
    "query_feats",
    "query_geom",
)
SUPPORTED_SUFFIXES = (".npz", ".pt", ".pth")


def _as_tensor(value: Any) -> Tensor:
    if torch.is_tensor(value):
        tensor = value.detach().clone()
    else:
        tensor = torch.as_tensor(value)
    if tensor.dtype == torch.float64:
        return tensor.float()
    if tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
        return tensor.float()
    return tensor


def _load_bundle(path: Path) -> Dict[str, Tensor]:
    if path.suffix == ".npz":
        with np.load(path) as data:
            return {key: _as_tensor(data[key]) for key in data.files}
    if path.suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path} must contain a dict-like tensor bundle")
        return {str(key): _as_tensor(value) for key, value in payload.items()}
    raise ValueError(f"unsupported sample file suffix: {path.suffix}")


def _list_sample_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for suffix in SUPPORTED_SUFFIXES:
        files.extend(root.glob(f"*{suffix}"))
    return sorted(files)


def _validate_keys(sample: Mapping[str, Tensor], source: Path) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in sample]
    if missing:
        raise ValueError(f"{source} is missing required keys: {missing}")


class LumiFileDataset(Dataset[Dict[str, Tensor]]):
    """Loads tensor samples from a file or a directory of files."""

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(self.data_path)

        self.files: List[Path] = []
        self.bundle: Dict[str, Tensor] | None = None
        if self.data_path.is_dir():
            self.files = _list_sample_files(self.data_path)
            if not self.files:
                raise ValueError(f"no sample files found under {self.data_path}")
        else:
            self.bundle = _load_bundle(self.data_path)
            _validate_keys(self.bundle, self.data_path)
            length = int(self.bundle[REQUIRED_KEYS[0]].size(0))
            for key in REQUIRED_KEYS:
                if int(self.bundle[key].size(0)) != length:
                    raise ValueError(f"{key} in {self.data_path} must share first dimension {length}")

    def __len__(self) -> int:
        if self.bundle is not None:
            return int(self.bundle[REQUIRED_KEYS[0]].size(0))
        return len(self.files)

    def _from_file(self, index: int) -> Dict[str, Tensor]:
        source = self.files[index]
        sample = _load_bundle(source)
        _validate_keys(sample, source)
        return {
            key: value
            for key, value in sample.items()
            if key in REQUIRED_KEYS or key in OPTIONAL_KEYS
        }

    def _from_bundle(self, index: int) -> Dict[str, Tensor]:
        if self.bundle is None:
            raise RuntimeError("bundle mode is not active")
        keys = list(REQUIRED_KEYS) + [key for key in OPTIONAL_KEYS if key in self.bundle]
        return {key: self.bundle[key][index] for key in keys}

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        if self.bundle is not None:
            return self._from_bundle(index)
        return self._from_file(index)


def lumi_collate(samples: Sequence[Mapping[str, Tensor]]) -> Dict[str, Tensor]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    keys = list(REQUIRED_KEYS)
    keys.extend(key for key in OPTIONAL_KEYS if all(key in sample for sample in samples))
    return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in keys}


def make_lumi_loader(
    data_path: str | Path,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader[Dict[str, Tensor]]:
    dataset = LumiFileDataset(data_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lumi_collate,
    )


def move_batch(batch: Mapping[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def build_meta(batch: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    meta_keys = (
        "coords",
        "point_geom",
        "point_boundary",
        "point_source",
        "point_grad",
        "shade",
        "blind",
        "light",
        "extra_control",
        "control_vec",
    )
    return {key: batch[key] for key in meta_keys if key in batch}


def available_keys() -> Iterable[str]:
    return tuple(REQUIRED_KEYS) + tuple(OPTIONAL_KEYS)
