#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

if ! command -v clang-format >/dev/null 2>&1; then
    echo "clang-format not found in PATH." >&2
    exit 1
fi

mapfile -t files < <(
    rg --files \
        -g '*.cpp' -g '*.hpp' -g '*.h' -g '*.cc' -g '*.cxx' -g '*.inl' \
        -g '!build/**' -g '!bin/**' -g '!runs/**' -g '!submissions/**'
)

if [[ ${#files[@]} -eq 0 ]]; then
    echo "No C/C++ files found to check."
    exit 0
fi

failed=0
for file in "${files[@]}"; do
    tmp=$(mktemp)
    clang-format "$file" > "$tmp"
    if ! cmp -s "$file" "$tmp"; then
        echo "Needs format: $file"
        failed=1
    fi
    rm -f "$tmp"
done

if [[ "$failed" -ne 0 ]]; then
    echo "Formatting check failed. Run scripts/format.sh" >&2
    exit 1
fi

echo "Formatting check passed."
