#!/bin/bash
#SBATCH --job-name=vf-diag
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/diag-%j.out
#SBATCH --error=logs/diag-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CKPT="${CKPT:-$ROOT/runs/test2/ckpt.pt}"
W="${W:-ema}"
echo "=== diagnose ===  job ${SLURM_JOB_ID:-local}  node $(hostname)"
echo "  CKPT : $CKPT"
echo "  W    : $W"
source "$ROOT/.venv/bin/activate"
cd "$ROOT"
python diagnose.py --ckpt "$CKPT" --weights "$W" --out "$ROOT/out/diagnose_$W.json"
