#!/bin/bash
#SBATCH --job-name=vf-uncond
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/uncond-%j.out
#SBATCH --error=logs/uncond-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CKPT="${CKPT:-$ROOT/runs/d_xs_dp02/ckpt.pt}"
echo "CKPT=$CKPT  node $(hostname)"
source "$ROOT/.venv/bin/activate"; cd "$ROOT"
python uncond.py --ckpt "$CKPT"
