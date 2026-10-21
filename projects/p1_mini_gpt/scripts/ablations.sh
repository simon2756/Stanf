#!/usr/bin/env bash
# P1 experiment grid. Intended for a single GPU (each run takes minutes on an A10/L4-class card).
# Usage: bash projects/p1_mini_gpt/scripts/ablations.sh
set -euo pipefail

STEPS=${STEPS:-5000}
COMMON=(--max-steps "$STEPS" --batch-size 64 --compile)

uv run python -m mini_gpt.train --out-dir runs/p1/baseline   "${COMMON[@]}"
uv run python -m mini_gpt.train --out-dir runs/p1/muon       "${COMMON[@]}" --optimizer muon
uv run python -m mini_gpt.train --out-dir runs/p1/cosine     "${COMMON[@]}" --schedule cosine
uv run python -m mini_gpt.train --out-dir runs/p1/learned-pos "${COMMON[@]}" --model.pos-emb learned
uv run python -m mini_gpt.train --out-dir runs/p1/no-qk-norm "${COMMON[@]}" --model.no-qk-norm
uv run python -m mini_gpt.train --out-dir runs/p1/gqa        "${COMMON[@]}" --model.n-kv-head 2

uv run python -m mini_gpt.report runs/p1
