#!/usr/bin/env bash
#SBATCH -J pl_commit_train
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/pl_commit_train_%j.out
#SBATCH -e logs/pl_commit_train_%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
VENV_DIR="${VENV_DIR:?VENV_DIR is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
INPUT_DIRS="${INPUT_DIRS:?INPUT_DIRS is required}"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"
arguments=()
IFS=':' read -ra roots <<< "$INPUT_DIRS"
if [[ "${#roots[@]}" -lt 1 ]]; then
  echo "INPUT_DIRS did not contain any dataset directories" >&2
  exit 2
fi
for root in "${roots[@]}"; do
  if [[ ! -d "$root" ]]; then
    echo "commitment-safety input directory does not exist: $root" >&2
    exit 2
  fi
  arguments+=(--input-dir "$root")
done
PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/scripts" "$VENV_DIR/bin/python" \
  scripts/train_commitment_safety_predictor.py \
  "${arguments[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --target sequence_safe \
  --max-observed-unsafe-rate 0 \
  --minimum-selected "${MINIMUM_SELECTED:-8}" \
  --seed "${SEED:-311}"
