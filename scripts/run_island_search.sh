#!/usr/bin/env bash
set -euo pipefail

BASE=${1:-submission.csv}
OUTROOT=${2:-runs/island_$(date +%Y%m%d_%H%M%S)}
JOBS=${JOBS:-24}

SEEDS_A=${SEEDS_A:-1..2000}
SEEDS_B=${SEEDS_B:-2001..4000}
SEEDS_C=${SEEDS_C:-4001..6000}

TARGET_A=${TARGET_A:-80,200}
TARGET_B=${TARGET_B:-120,200}
TARGET_C=${TARGET_C:-60,200}

mkdir -p "$OUTROOT"

echo "Base: $BASE"
echo "Out: $OUTROOT"
echo "Jobs: $JOBS"
echo "Seeds A: $SEEDS_A  Target A: $TARGET_A"
echo "Seeds B: $SEEDS_B  Target B: $TARGET_B"
echo "Seeds C: $SEEDS_C  Target C: $TARGET_C"

# Config A: broad high-n range
PYTHONPATH=$PWD python -m santa_packing._tools.hunt_compact_contact \
  --base "$BASE" \
  --out-dir "$OUTROOT/highn_a" \
  --seeds "$SEEDS_A" \
  --jobs "$JOBS" \
  --target-range "$TARGET_A" \
  --passes 4 \
  --attempts-per-pass 8 \
  --patience 3 \
  --boundary-topk 12 \
  --push-bisect-iters 10 \
  --push-max-step-frac 0.15 \
  --plateau-eps 1e-6 \
  --diag-frac 0.15 \
  --diag-rand 0.4 \
  --center-bias 0.6 \
  --interior-prob 0.1 \
  --shake-pos 0.01 \
  --shake-rot-deg 0.25 \
  --shake-prob 0.35 \
  --smooth-window 0

# Config B: focus late n for private stability
PYTHONPATH=$PWD python -m santa_packing._tools.hunt_compact_contact \
  --base "$BASE" \
  --out-dir "$OUTROOT/highn_b" \
  --seeds "$SEEDS_B" \
  --jobs "$JOBS" \
  --target-range "$TARGET_B" \
  --passes 5 \
  --attempts-per-pass 10 \
  --patience 4 \
  --boundary-topk 16 \
  --push-bisect-iters 12 \
  --push-max-step-frac 0.2 \
  --plateau-eps 1e-6 \
  --diag-frac 0.2 \
  --diag-rand 0.5 \
  --center-bias 0.5 \
  --interior-prob 0.15 \
  --shake-pos 0.015 \
  --shake-rot-deg 0.5 \
  --shake-prob 0.4 \
  --smooth-window 0

# Config C: mid/high blend
PYTHONPATH=$PWD python -m santa_packing._tools.hunt_compact_contact \
  --base "$BASE" \
  --out-dir "$OUTROOT/highn_c" \
  --seeds "$SEEDS_C" \
  --jobs "$JOBS" \
  --target-range "$TARGET_C" \
  --passes 4 \
  --attempts-per-pass 8 \
  --patience 3 \
  --boundary-topk 12 \
  --push-bisect-iters 10 \
  --push-max-step-frac 0.15 \
  --plateau-eps 1e-6 \
  --diag-frac 0.15 \
  --diag-rand 0.4 \
  --center-bias 0.6 \
  --interior-prob 0.1 \
  --shake-pos 0.01 \
  --shake-rot-deg 0.25 \
  --shake-prob 0.35 \
  --smooth-window 0

# Merge best per-n
PYTHONPATH=$PWD python scripts/merge_submissions_best.py \
  --base "$BASE" \
  --out "$OUTROOT/merged_best.csv" \
  "$OUTROOT/highn_a/ensemble.csv" \
  "$OUTROOT/highn_b/ensemble.csv" \
  "$OUTROOT/highn_c/ensemble.csv"

# Score without overlap checks (fast)
PYTHONPATH=$PWD python -m santa_packing.cli.score_submission "$OUTROOT/merged_best.csv" --no-overlap --pretty > "$OUTROOT/merged_best_score.json"

echo "Done. See $OUTROOT/merged_best.csv"
