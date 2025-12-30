# Santa 2025 - Christmas Tree Packing Challenge (Python/JAX)

This repository contains a Python/JAX implementation for the Santa 2025 Kaggle Challenge, replacing the previous C++ codebase.
It uses GPU-accelerated Batch Simulated Annealing to optimize tree packing.

## Project Structure
- `src/`: Source code (JAX optimization).
- `scripts/`: Utility scripts.
- `runs/`: Output directory.

## Requirements
- Python 3.10+
- NVIDIA GPU (Recommended) with CUDA
- JAX, Matplotlib, NumPy

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install jax jaxlib matplotlib numpy
```

## Running the Optimization

To run the JAX-based Batch Simulated Annealing:
```bash
python src/main.py --n_trees 25 --batch_size 128 --n_steps 1000
```

- `--n_trees`: Number of trees to pack.
- `--batch_size`: Number of parallel optimization chains (higher = better exploration, more GPU memory).
- `--n_steps`: Number of SA iterations.

## Visualization
The script produces `best_packing.png` visualizing the result.

## Submission + Scoring (Python)

Generate a hybrid submission (SA for small N, lattice for large N):
```bash
python3 scripts/generate_submission.py --out submission.csv --nmax 200
```

Score a submission locally (JSON output):
```bash
python3 scripts/score_submission.py submission.csv --pretty
```
