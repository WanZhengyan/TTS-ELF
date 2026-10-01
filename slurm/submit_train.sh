#!/bin/bash
#SBATCH --job-name=vf-train
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Derived values are echoed before use: a var set after the arg string is built
# would be silently ignored, so the log must show what actually ran.
MODEL="${MODEL:-ELF-T}"
STEPS="${STEPS:-2000}"
BATCH="${BATCH:-16}"
LR="${LR:-3e-4}"
OVERFIT="${OVERFIT:-0}"
EXTRA="${EXTRA:-}"
RUN="${RUN:-test1}"

echo "=== voiceflow train ==="
echo "  job     : ${SLURM_JOB_ID:-local}   node $(hostname)"
echo "  MODEL   : $MODEL"
echo "  STEPS   : $STEPS"
echo "  BATCH   : $BATCH"
echo "  LR      : $LR"
echo "  OVERFIT : $OVERFIT"
echo "  EXTRA   : $EXTRA"
echo "  RUN     : $RUN -> $ROOT/runs/$RUN"
echo "  started : $(date -Is)"

source "$ROOT/.venv/bin/activate"
cd "$ROOT"
ARGS="--model $MODEL --steps $STEPS --batch $BATCH --lr $LR --out $ROOT/runs/$RUN"
if [ "$OVERFIT" != "0" ]; then ARGS="$ARGS --overfit $OVERFIT"; fi
if [ -n "$EXTRA" ]; then ARGS="$ARGS $EXTRA"; fi
echo "  cmd     : python train.py $ARGS"
python train.py $ARGS
echo "  finished: $(date -Is)"
