# Latent_lumi

## Setup

```bash
cd Latent_lumi
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Run Synthetic Training

```bash
python scripts/run_synthetic_demo.py --steps 30
```

The demo runs without external data and prints loss values, prediction shape, and gate statistics.

## Run File Training

```bash
python scripts/train_from_files.py --data path/to/data_dir --epochs 5
```

The data path can be a directory of `.npz`, `.pt`, or `.pth` sample files, or one `.npz/.pt/.pth` file whose arrays share the same first dimension.

Required keys are `history`, `coords`, `query_coords`, and `target`.

## Run Tests

```bash
pytest -q
```
