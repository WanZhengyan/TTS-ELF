#!/bin/bash
#SBATCH --job-name=vf-prep
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/prep-%j.out
#SBATCH --error=logs/prep-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LIMIT="${LIMIT:-0}"
echo "=== preprocess ===  job ${SLURM_JOB_ID:-local}  node $(hostname)  LIMIT=$LIMIT  $(date -Is)"
source "$ROOT/.venv/bin/activate"
cd "$ROOT"
python preprocess.py --limit "$LIMIT"
echo "finished $(date -Is)"
