#!/bin/bash
#SBATCH --job-name=vf-spkcpu
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/spkcpu-%j.out
#SBATCH --error=logs/spkcpu-%j.err
# x-vectors need no GPU: UniSpeechSat-base on 3 s of audio runs on the CPU
# cores a GPU node leaves idle, without waiting for a GPU to free up.
set -uo pipefail
R="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CACHE="${CACHE:?}"; SPLIT="${SPLIT:?}"
echo "=== x-vector on CPU: $CACHE ($SPLIT)  job ${SLURM_JOB_ID:-local} $(hostname) $(date -Is)"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export CUDA_VISIBLE_DEVICES=""
source "$R/.venv/bin/activate"; cd "$R"
python preprocess_spk.py --cache "$CACHE" --split "$SPLIT" --prompt_sec 3.0 --batch 32 || exit 1
echo "done $(date -Is)"
