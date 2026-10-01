#!/bin/bash
#SBATCH --job-name=vf-test0
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=logs/test0-%j.out
#SBATCH --error=logs/test0-%j.err

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N_CLIPS="${N_CLIPS:-500}"
SEED="${SEED:-0}"
OUT_DIR="${OUT_DIR:-$ROOT/out}"

# Echo every derived value before use — a shell var set after a config string is
# built is silently ignored, so the log has to show what actually ran.
echo "=== voiceflow Test 0 ==="
echo "  job          : ${SLURM_JOB_ID:-local}"
echo "  node         : $(hostname)"
echo "  ROOT         : $ROOT"
echo "  N_CLIPS      : $N_CLIPS"
echo "  SEED         : $SEED"
echo "  OUT_DIR      : $OUT_DIR"
echo "  started      : $(date -Is)"

source "$ROOT/.venv/bin/activate"
echo "  python       : $(which python)"
python -c "import torch;print('  torch        :',torch.__version__,'| cuda',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$ROOT"
python probe_test0.py \
  --n_clips "$N_CLIPS" \
  --seed "$SEED" \
  --out_dir "$OUT_DIR"

echo "  finished     : $(date -Is)"
