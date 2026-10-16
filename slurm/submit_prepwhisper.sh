#!/bin/bash
#SBATCH --job-name=vf-prepwh
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/prepwh-%j.out
#SBATCH --error=logs/prepwh-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CACHE="${CACHE:-$ROOT/cache_lj_uni}"
MODEL="${MODEL:-openai/whisper-small}"
SPLIT="${SPLIT:-train-clean-100}"   # 必须和 cache 的 split 一致，否则 flac 路径取错
echo "=== frozen Whisper features -> cache ==="
echo "  job   : ${SLURM_JOB_ID:-local}   node: $(hostname)"
echo "  CACHE : $CACHE"
echo "  MODEL : $MODEL"
echo "  SPLIT : $SPLIT"
echo "  SUFFIX: ${SUFFIX:-(无)}"
echo "  start : $(date -Is)"
export TMPDIR="$ROOT/tmp"; mkdir -p "$TMPDIR"
source "$ROOT/.venv/bin/activate"
cd "$ROOT"
python preprocess_whisper.py --cache "$CACHE" --model "$MODEL" --split "$SPLIT" --suffix "${SUFFIX:-}"
echo "  done  : $(date -Is)"
