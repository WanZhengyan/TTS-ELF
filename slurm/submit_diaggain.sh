#!/bin/bash
#SBATCH --job-name=vf-diaggain
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/diaggain-%j.out
#SBATCH --error=logs/diaggain-%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CACHE="${CACHE:-$ROOT/cache_libri_train_clean_100}"
OUT="${OUT:-$ROOT/runs/uni}"
STEPS="${STEPS:-40000}"
BATCH="${BATCH:-16}"
MAXLEN="${MAXLEN:-512}"
MAXTEXT="${MAXTEXT:-48}"
PS2T="${PS2T:-0.5}"
LR="${LR:-1e-4}"
EVERY="${EVERY:-2000}"

# Echo every derived value: a var set after the config string is built is
# silently ignored, so the log has to show what actually ran.
echo "=== unified TTS + S2T ==="
echo "  job     : ${SLURM_JOB_ID:-local}   node: $(hostname)"
echo "  CACHE   : $CACHE"
echo "  OUT     : $OUT"
echo "  STEPS   : $STEPS   BATCH: $BATCH   LR: $LR"
echo "  MAXLEN  : $MAXLEN   MAXTEXT: $MAXTEXT   P_S2T: $PS2T"
echo "  EVERY   : $EVERY"
echo "  started : $(date -Is)"

export TMPDIR="$ROOT/tmp"
mkdir -p "$TMPDIR"
source "$ROOT/.venv/bin/activate"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$ROOT"
python diag_gain.py

echo "  finished : $(date -Is)"
