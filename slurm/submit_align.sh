#!/bin/bash
#SBATCH --job-name=vf-align
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/align-%j.out
#SBATCH --error=logs/align-%j.err
set -euo pipefail
R="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LIMIT="${LIMIT:-0}"; OUT="${OUT:-$R/cache/align.pt}"
echo "LIMIT=$LIMIT OUT=$OUT node $(hostname)"
source "$R/.venv/bin/activate"; cd "$R"
python align_text.py --limit "$LIMIT" --batch 16 --out "$OUT"
