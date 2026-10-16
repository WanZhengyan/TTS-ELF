#!/bin/bash
#SBATCH --job-name=vf-why
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/why-%j.out
#SBATCH --error=logs/why-%j.err
set -uo pipefail
R="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TMPDIR="$R/tmp"; mkdir -p "$TMPDIR"
source "$R/.venv/bin/activate"; cd "$R"
python why_lj_works.py --n_wav "${NWAV:-4}" --align_n "${ALIGNN:-30}"
