#!/bin/bash
#SBATCH --job-name=vf-ttscfg
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/ttscfg-%j.out
#SBATCH --error=logs/ttscfg-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN="${RUN:-p_dp5}"
CACHE="${CACHE:-$ROOT/cache_lj_uni}"
N="${N:-40}"
NFE="${NFE:-32}"
WS="${WS:-1.0 1.5 2.0 3.0}"
echo "=== TTS voice-channel CFG sweep (inference only) ==="
echo "  job  : ${SLURM_JOB_ID:-local}   node: $(hostname)"
echo "  RUN  : $RUN    N: $N   NFE: $NFE"
echo "  CACHE_SPLIT: ${SPLIT:-test}   LIBRI_SPLIT(找 flac 用): ${LIBRISPLIT:-train-clean-100}"
echo "  WS   : $WS"
echo "  NFE: ${NFE:-32}  GAMMA: ${GAMMA:-0.0} (SDE churn，针对 exposure gap)"
echo "  PROMPT: ${PROMPT:-0.0}  (voice prompt 占 clip 的比例；训练用的是 0-0.3)"
echo "  start: $(date -Is)"
export TMPDIR="$ROOT/tmp"; mkdir -p "$TMPDIR"
source "$ROOT/.venv/bin/activate"; cd "$ROOT"
for w in $WS; do
  echo ""; echo "################ cfg_w = $w ################"
  python sample_unified.py --ckpt "$ROOT/runs/$RUN/ckpt.pt" --cache "$CACHE" \
    --n "$N" --nfe "${NFE}" --gamma "${GAMMA:-0.0}" --prompt "${PROMPT:-0.0}" --cfg_w "$w" --cache_split "${SPLIT:-test}" --split "${LIBRISPLIT:-train-clean-100}" \
    --out "$ROOT/out/${RUN}_cfgw${w}.json" \
    --page "$ROOT/listen_${RUN}_cfgw${w}.html"
done
echo ""; echo "  done : $(date -Is)"
