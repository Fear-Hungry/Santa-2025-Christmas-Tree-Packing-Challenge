#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
BIN_DIR="$ROOT_DIR/bin"
RUNS_DIR="$ROOT_DIR/runs/regression"

EXPECTED_FILE_DEFAULT="$ROOT_DIR/scripts/regression_expected.env"
EXPECTED_FILE="${EXPECTED_FILE:-$EXPECTED_FILE_DEFAULT}"
TOL="${TOL:-1e-6}"
RECORD=0

EXPECTED_SOLVER_BASELINE=""
EXPECTED_SOLVER_TILE_SEED1=""
EXPECTED_SOLVER_TILE_SEED2=""
GOT_SOLVER_BASELINE=""
GOT_SOLVER_TILE_SEED1=""
GOT_SOLVER_TILE_SEED2=""

usage() {
    cat <<EOF
Usage: $0 [--record] [--expected <file>] [--tol <value>]
  --record          Run and write expected scores to <file>.
  --expected <file> Read/write expected scores file (default: scripts/regression_expected.env).
  --tol <value>     Absolute tolerance for score compare (default: 1e-6).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --record)
            RECORD=1
            ;;
        --expected)
            EXPECTED_FILE="$2"
            shift
            ;;
        --tol)
            TOL="$2"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown arg: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

if [[ -f "$EXPECTED_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$EXPECTED_FILE"
fi

if [[ "$RECORD" != "1" ]]; then
    if [[ -z "$EXPECTED_SOLVER_BASELINE" || -z "$EXPECTED_SOLVER_TILE_SEED1" || -z "$EXPECTED_SOLVER_TILE_SEED2" ]]; then
        echo "Expected scores not set. Run with --record to create $EXPECTED_FILE." >&2
        exit 1
    fi
fi

mkdir -p "$RUNS_DIR"

score_file() {
    local name="$1"
    echo "$RUNS_DIR/${name}.score.txt"
}

extract_score() {
    local file="$1"
    awk '/^Score: /{print $2; exit}' "$file"
}

compare_score() {
    local expected="$1"
    local got="$2"
    local tol="$3"

    if [[ -z "$expected" ]]; then
        echo "Expected score not set." >&2
        return 1
    fi

    local ok
    ok=$(awk -v a="$expected" -v b="$got" -v t="$tol" 'BEGIN{d=a-b; if (d<0) d=-d; print (d<=t) ? 1 : 0}')
    if [[ "$ok" != "1" ]]; then
        echo "Score mismatch: expected=$expected got=$got tol=$tol" >&2
        return 1
    fi
}

run_case() {
    local name="$1"
    local expected="$2"
    shift 2

    local csv="$RUNS_DIR/${name}.csv"
    local score_out
    score_out=$(score_file "$name")

    echo "Running: $* --output $csv"
    "$@" --output "$csv"
    "$BIN_DIR/score_submission" "$csv" --breakdown > "$score_out"

    local score
    score=$(extract_score "$score_out")
    if [[ -z "$score" ]]; then
        echo "Could not parse score for $name." >&2
        return 1
    fi

    case "$name" in
        solver_baseline)
            GOT_SOLVER_BASELINE="$score"
            ;;
        solver_tile_seed1)
            GOT_SOLVER_TILE_SEED1="$score"
            ;;
        solver_tile_seed2)
            GOT_SOLVER_TILE_SEED2="$score"
            ;;
    esac

    if [[ "$RECORD" == "1" ]]; then
        echo "RECORDED: $name score=$score"
        return 0
    fi

    compare_score "$expected" "$score" "$TOL"
    echo "OK: $name score=$score"
}

run_case "solver_baseline" "$EXPECTED_SOLVER_BASELINE" \
    "$BIN_DIR/solver_baseline"

run_case "solver_tile_seed1" "$EXPECTED_SOLVER_TILE_SEED1" \
    "$BIN_DIR/solver_tile" --seed 1 --k 2 --tile-iters 200

run_case "solver_tile_seed2" "$EXPECTED_SOLVER_TILE_SEED2" \
    "$BIN_DIR/solver_tile" --seed 2 --k 2 --tile-iters 200

if [[ "$RECORD" == "1" ]]; then
    if [[ -z "$GOT_SOLVER_BASELINE" || -z "$GOT_SOLVER_TILE_SEED1" || -z "$GOT_SOLVER_TILE_SEED2" ]]; then
        echo "Missing recorded scores; expected file not written." >&2
        exit 1
    fi
    cat > "$EXPECTED_FILE" <<EOF
EXPECTED_SOLVER_BASELINE=$GOT_SOLVER_BASELINE
EXPECTED_SOLVER_TILE_SEED1=$GOT_SOLVER_TILE_SEED1
EXPECTED_SOLVER_TILE_SEED2=$GOT_SOLVER_TILE_SEED2
EOF
    echo "Wrote expected scores to $EXPECTED_FILE"
    exit 0
fi
