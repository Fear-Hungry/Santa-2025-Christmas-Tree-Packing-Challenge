#!/usr/bin/env bash
set -euo pipefail

BASE=${1:-submission.csv}
TARGETS_FILE=${2:-runs/targets_highn_top40.txt}
OUTROOT=${3:-runs/targeted_$(date +%Y%m%d_%H%M%S)}
JOBS=${JOBS:-24}
SEEDS=${SEEDS:-1..2000}

if [[ ! -f "$TARGETS_FILE" ]]; then
  echo "Targets file not found: $TARGETS_FILE" >&2
  exit 1
fi

mkdir -p "$OUTROOT"

echo "Base: $BASE"
echo "Targets: $TARGETS_FILE"
echo "Out: $OUTROOT"
echo "Jobs: $JOBS"
echo "Seeds: $SEEDS"

# Run compact_contact per target N (serial over N, parallel over jobs).
while read -r N; do
  [[ -z "$N" ]] && continue
  echo "[target n=$N]"
  PYTHONPATH=$PWD python -m santa_packing._tools.hunt_compact_contact \
    --base "$BASE" \
    --out-dir "$OUTROOT/n${N}" \
    --seeds "$SEEDS" \
    --jobs "$JOBS" \
    --target-range "$N,$N" \
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

done < "$TARGETS_FILE"

# Merge all ensembles by best per-n
ENSEMBLES=()
for d in "$OUTROOT"/n*/ensemble.csv; do
  [[ -f "$d" ]] && ENSEMBLES+=("$d")
 done

if [[ ${#ENSEMBLES[@]} -gt 0 ]]; then
  PYTHONPATH=$PWD python scripts/merge_submissions_best.py \
    --base "$BASE" \
    --out "$OUTROOT/merged_best.csv" \
    "${ENSEMBLES[@]}"

  PYTHONPATH=$PWD python -m santa_packing.cli.score_submission "$OUTROOT/merged_best.csv" --no-overlap --pretty > "$OUTROOT/merged_best_score.json"
  echo "Done. See $OUTROOT/merged_best.csv"
else
  echo "No ensemble outputs found; nothing to merge." >&2
  exit 1
fi
