#!/bin/bash
#SBATCH --job-name=vf-probe
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/probe-%j.out
#SBATCH --error=logs/probe-%j.err
set -uo pipefail
R="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TMPDIR="$R/tmp"; mkdir -p "$TMPDIR"
source "$R/.venv/bin/activate"; cd "$R"
python "${SCRIPT:-probe_sccfg.py}"
