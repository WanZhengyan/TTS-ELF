#!/bin/bash
#SBATCH --job-name=vf-abl
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/abl-%j.out
#SBATCH --error=logs/abl-%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NWER="${NWER:-40}"
NWAV="${NWAV:-3}"
NFE="${NFE:-32}"

echo "=== all checkpoints, both directions, one page ==="
echo "  job  : ${SLURM_JOB_ID:-local}   node: $(hostname)"
echo "  NWER : $NWER   NWAV: $NWAV   NFE: $NFE"
echo "  start: $(date -Is)"

export TMPDIR="$ROOT/tmp"
mkdir -p "$TMPDIR"
source "$ROOT/.venv/bin/activate"
cd "$ROOT"
python ablate_sampling.py --ckpt "${ACKPT:-$ROOT/runs/hy_wh_af/ckpt.pt}" --n_batch "${NB:-3}" --acfg "${ACFG:-0}" --out "${AOUT:-$ROOT/out/ablate_sampling.json}"
echo "  done : $(date -Is)"
