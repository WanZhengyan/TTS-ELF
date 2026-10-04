#!/bin/bash
#SBATCH --job-name=vf-ts
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=logs/ts-%j.out
#SBATCH --error=logs/ts-%j.err
set -euo pipefail
R="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$R/.venv/bin/activate"; cd "$R"; python text_sens.py
