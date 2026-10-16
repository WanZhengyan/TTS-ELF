#!/bin/bash
#SBATCH --job-name=vf-sampuni
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/sampuni-%j.out
#SBATCH --error=logs/sampuni-%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CKPT="${CKPT:-$ROOT/runs/ch_tts/ckpt.pt}"
CACHE="${CACHE:-$ROOT/cache_libri_train_clean_100}"
N="${N:-50}"
NFE="${NFE:-32}"
GAMMA="${GAMMA:-0.0}"
OUT="${OUT:-$ROOT/out/unified_wer.json}"
PAGE="${PAGE:-$ROOT/listen_unified.html}"

echo "=== unified TTS -> Vocos -> WER/UTMOS ==="
echo "  job   : ${SLURM_JOB_ID:-local}   node: $(hostname)"
echo "  CKPT  : $CKPT"
echo "  CACHE : $CACHE"
echo "  N     : $N   NFE: $NFE   GAMMA: $GAMMA"
echo "  OUT   : $OUT"
echo "  PAGE  : $PAGE"
echo "  start : $(date -Is)"

export TMPDIR="$ROOT/tmp"
mkdir -p "$TMPDIR"
source "$ROOT/.venv/bin/activate"
cd "$ROOT"
python sample_unified.py --ckpt "$CKPT" --cache "$CACHE" --n "$N" --nfe "$NFE" \
  --gamma "$GAMMA" --out "$OUT" --page "$PAGE" --prompt "${PROMPT:-0.0}" \
  --cfg_w "${CFGW:-1.0}" --cache_split "${SPLIT:-test}" --split "${LIBRISPLIT:-train-clean-100}"

echo "  done  : $(date -Is)"
