#!/bin/bash
#SBATCH --job-name=vf-cfgvar
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/cfgvar-%j.out
#SBATCH --error=logs/cfgvar-%j.err
set -euo pipefail
R="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
N="${N:-40}"
echo "=== TTS CFG：先量采样方差，再读引导效应 ==="
echo "  job  : ${SLURM_JOB_ID:-local}   node: $(hostname)   N: $N"
echo "  A 部分: p_dp5  w={1.0,2.0} x seed={0,1,2}  -- 同一 w 的三个种子给出方差"
echo "  B 部分: hy_wh_af w={1.0,1.5,2.0} seed=0    -- 基线稳定的 ckpt 上独立验证"
echo "  同一 seed 下不同 w 共用同一组初始噪声，所以是配对比较"
echo "  start: $(date -Is)"
export TMPDIR="$R/tmp"; mkdir -p "$TMPDIR"
source "$R/.venv/bin/activate"; cd "$R"
run(){ # ckpt w seed
  echo ""; echo "######## $1  w=$2  seed=$3 ########"
  python sample_unified.py --ckpt "$R/runs/$1/ckpt.pt" --cache "$R/cache_lj_uni" \
    --n "$N" --nfe 32 --gamma 0.0 --prompt 0.0 --cfg_w "$2" --seed "$3" \
    --out "$R/out/var_$1_w$2_s$3.json" --page "$R/listen_var_$1_w$2_s$3.html"
}
for s in 0 1 2; do for w in 1.0 2.0; do run p_dp5 $w $s; done; done
for w in 1.0 1.5 2.0; do run hy_wh_af $w 0; done
echo ""; echo "  done : $(date -Is)"
