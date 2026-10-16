#!/bin/bash
#SBATCH --job-name=vf-prepspk
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/prepspk-%j.out
#SBATCH --error=logs/prepspk-%j.err
set -uo pipefail
R="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CACHE="${CACHE:-$R/cache_libri_train_clean_100}"
SPLIT="${SPLIT:-train-clean-100}"
echo "=== x-vector 说话人嵌入 -> cache ==="
echo "  job : ${SLURM_JOB_ID:-local}  node: $(hostname)"
echo "  CACHE : $CACHE"
echo "  SPLIT : $SPLIT"
echo "  MODEL : ${MODEL:-microsoft/unispeech-sat-base-plus-sv}"
echo "  start : $(date -Is)"
export TMPDIR="$R/tmp"; mkdir -p "$TMPDIR"
source "$R/.venv/bin/activate"; cd "$R"
python preprocess_spk.py --cache "$CACHE" --split "$SPLIT" \
  --model "${MODEL:-microsoft/unispeech-sat-base-plus-sv}" \
  --prompt_sec "${PSEC:-3.0}"
echo "  done : $(date -Is)"
