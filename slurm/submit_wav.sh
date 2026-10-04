#!/bin/bash
#SBATCH --job-name=vf-wav
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/wav-%j.out
#SBATCH --error=logs/wav-%j.err
set -euo pipefail
R="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CKPT="${CKPT:-$R/runs/test2/ckpt.pt}"
PAGE="${PAGE:-$R/listen.html}"
ODIR="${ODIR:-$R/out/listen}"
echo "CKPT=$CKPT  PAGE=$PAGE  ODIR=$ODIR"
source "$R/.venv/bin/activate"; cd "$R"
python sample_wav.py --ckpt "$CKPT" --page "$PAGE" --out_dir "$ODIR" --n 6
