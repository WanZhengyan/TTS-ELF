#!/bin/bash
#SBATCH --job-name=vf-s2tstep
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/s2tstep-%j.out
#SBATCH --error=logs/s2tstep-%j.err
set -uo pipefail
R="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TMPDIR="$R/tmp"; mkdir -p "$TMPDIR"
source "$R/.venv/bin/activate"; cd "$R"
python probe_s2t_steps.py --n "${N:-400}" --steps "${STEPS:-1,2,4,8,32,128}"
