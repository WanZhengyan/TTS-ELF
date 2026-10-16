#!/bin/bash
#SBATCH --job-name=vf-prov
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/prov-%j.out
#SBATCH --error=logs/prov-%j.err

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
python provenance.py
echo "  done : $(date -Is)"
