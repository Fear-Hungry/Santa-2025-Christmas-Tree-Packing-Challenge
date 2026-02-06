#!/usr/bin/env bash
set -euo pipefail

BASE=${1:-submission.csv}
OUTROOT=${2:-runs/sweep_$(date +%Y%m%d_%H%M%S)}
JOBS=${JOBS:-24}
SEEDS=${SEEDS:-1..2000}
TARGET=${TARGET:-80,200}

mkdir -p "$OUTROOT"

echo "Base: $BASE"
echo "Out: $OUTROOT"
echo "Jobs: $JOBS"
echo "Seeds: $SEEDS"
echo "Target: $TARGET"

run_cfg() {
  local name=$1
  shift
  PYTHONPATH=$PWD python -m santa_packing._tools.hunt_compact_contact \
    --base "$BASE" \
    --out-dir "$OUTROOT/$name" \
    --seeds "$SEEDS" \
    --jobs "$JOBS" \
    --target-range "$TARGET" \
    --passes 5 \
    --attempts-per-pass 10 \
    --patience 4 \
    --plateau-eps 1e-6 \
    --smooth-window 0 \
    "$@"
}

# Config A: aggressive boundary + push
run_cfg cfg_a \
  --boundary-topk 20 \
  --push-bisect-iters 12 \
  --push-max-step-frac 0.25 \
  --diag-frac 0.2 \
  --diag-rand 0.5 \
  --center-bias 0.5 \
  --interior-prob 0.15 \
  --shake-pos 0.02 \
  --shake-rot-deg 0.6 \
  --shake-prob 0.45

# Config B: interior bias
run_cfg cfg_b \
  --boundary-topk 12 \
  --push-bisect-iters 10 \
  --push-max-step-frac 0.18 \
  --diag-frac 0.15 \
  --diag-rand 0.4 \
  --center-bias 0.35 \
  --interior-prob 0.3 \
  --shake-pos 0.015 \
  --shake-rot-deg 0.4 \
  --shake-prob 0.4

# Config C: rotation-heavy
run_cfg cfg_c \
  --boundary-topk 16 \
  --push-bisect-iters 12 \
  --push-max-step-frac 0.2 \
  --diag-frac 0.25 \
  --diag-rand 0.6 \
  --center-bias 0.55 \
  --interior-prob 0.1 \
  --shake-pos 0.01 \
  --shake-rot-deg 1.0 \
  --shake-prob 0.5

# Merge by best per-n with s_max guard
PYTHONPATH=$PWD python scripts/merge_submissions_best.py \
  --base "$BASE" \
  --out "$OUTROOT/merged_best.csv" \
  --max-smax-delta 0.0 \
  "$OUTROOT/cfg_a/ensemble.csv" \
  "$OUTROOT/cfg_b/ensemble.csv" \
  "$OUTROOT/cfg_c/ensemble.csv"

PYTHONPATH=$PWD python -m santa_packing.cli.score_submission "$OUTROOT/merged_best.csv" --no-overlap --pretty > "$OUTROOT/merged_best_score.json"

echo "Done. See $OUTROOT/merged_best.csv"
