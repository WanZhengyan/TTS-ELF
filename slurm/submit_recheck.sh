#!/bin/bash
#SBATCH --job-name=vf-recheck
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/recheck-%j.out
#SBATCH --error=logs/recheck-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUNS="${RUNS:-p_dp5_acfg hy_wh_af}"
N="${N:-40}"
echo "=== pre-fix ckpt 在当前 collate 下重测 TTS (cfg_w=1.0, 无引导) ==="
echo "  job  : ${SLURM_JOB_ID:-local}   node: $(hostname)"
echo "  RUNS : $RUNS    N: $N"
echo "  目的 : 检验 '10-10 17:04 的 collate 改动让所有 ckpt 的 TTS 数字跳一档' 这个解释"
echo "  start: $(date -Is)"
export TMPDIR="$ROOT/tmp"; mkdir -p "$TMPDIR"
source "$ROOT/.venv/bin/activate"; cd "$ROOT"
for r in $RUNS; do
  echo ""; echo "################ $r ################"
  python sample_unified.py --ckpt "$ROOT/runs/$r/ckpt.pt" --cache "$ROOT/cache_lj_uni" \
    --n "$N" --nfe 32 --gamma 0.0 --prompt 0.0 --cfg_w 1.0 \
    --out "$ROOT/out/${r}_recheck.json" --page "$ROOT/listen_${r}_recheck.html"
done
echo ""; echo "  done : $(date -Is)"
