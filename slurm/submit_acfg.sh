#!/bin/bash
#SBATCH --job-name=vf-acfg
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/acfg-%j.out
#SBATCH --error=logs/acfg-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUNS="${RUNS:-p_dp5_acfg,p_dp5}"
W="${W:-1.0,1.5,2.0,3.0}"
N="${N:-288}"
NFE="${NFE:-32}"
echo "=== audio classifier-free guidance, inference only ==="
echo "  job  : ${SLURM_JOB_ID:-local}   node: $(hostname)"
echo "  RUNS : $RUNS"
echo "  W    : $W     N: $N    NFE: $NFE"
echo "  start: $(date -Is)"
export TMPDIR="$ROOT/tmp"
mkdir -p "$TMPDIR"
source "$ROOT/.venv/bin/activate"
cd "$ROOT"
python acfg_sweep.py --runs "$RUNS" --w "$W" --n "$N" --nfe "$NFE" --cache "${CACHE:-$ROOT/cache_lj_uni}" --split "${SPLIT:-test}"
echo "  done : $(date -Is)"
