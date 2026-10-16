#!/bin/bash
#SBATCH --job-name=vf-unitrain
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/unitrain-%j.out
#SBATCH --error=logs/unitrain-%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CACHE="${CACHE:-$ROOT/cache_libri_test_clean}"
STEPS="${STEPS:-50}"
BATCH="${BATCH:-8}"
MAXLEN="${MAXLEN:-512}"
PS2T="${PS2T:-0.5}"

echo "=== unified TTS + S2T smoke run ==="
echo "  job     : ${SLURM_JOB_ID:-local}   node: $(hostname)"
echo "  CACHE   : $CACHE"
echo "  STEPS   : $STEPS   BATCH: $BATCH   MAXLEN: $MAXLEN   P_S2T: $PS2T"
echo "  started : $(date -Is)"

source "$ROOT/.venv/bin/activate"
cd "$ROOT"
python train_unified.py --cache "$CACHE" --steps "$STEPS" --batch "$BATCH" \
  --max_length "$MAXLEN" --p_s2t "$PS2T"

echo "  finished : $(date -Is)"
