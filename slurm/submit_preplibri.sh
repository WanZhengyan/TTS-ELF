#!/bin/bash
#SBATCH --job-name=vf-preplibri
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/preplibri-%j.out
#SBATCH --error=logs/preplibri-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPLIT="${SPLIT:-test-clean}"
LIMIT="${LIMIT:-0}"

echo "=== LibriSpeech -> mel cache ==="
echo "  job          : ${SLURM_JOB_ID:-local}"
echo "  node         : $(hostname)"
echo "  ROOT         : $ROOT"
echo "  SPLIT        : $SPLIT"
echo "  LIMIT        : $LIMIT"
echo "  started      : $(date -Is)"

source "$ROOT/.venv/bin/activate"
cd "$ROOT"
python preprocess_libri.py --split "$SPLIT" --limit "$LIMIT"

echo "  finished     : $(date -Is)"
