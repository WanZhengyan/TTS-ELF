#!/bin/bash
#SBATCH --job-name=vf-sched
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/sched-%j.out
#SBATCH --error=logs/sched-%j.err
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "=== sched test ===  job ${SLURM_JOB_ID:-local}  node $(hostname)"
source "$ROOT/.venv/bin/activate"; cd "$ROOT"
python sched_test.py
