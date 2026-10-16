#!/bin/bash
#SBATCH --job-name=vf-prepmel
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/prepmel-%j.out
#SBATCH --error=logs/prepmel-%j.err
set -euo pipefail
R="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$R/.venv/bin/activate"; cd "$R"
python preprocess_mel.py --out "$R/cache_mel"
