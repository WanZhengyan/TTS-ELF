#!/bin/bash
#SBATCH --job-name=vf-proto
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/proto-%j.out
#SBATCH --error=logs/proto-%j.err
set -euo pipefail
R="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TMPDIR="$R/tmp"; mkdir -p "$TMPDIR"
source "$R/.venv/bin/activate"; cd "$R"
python eval_protocol.py --run "${RUN:-libri_tc100_uni_ddp}" \
  --prompt_sec "${PSEC:-3.0}" --tts_n "${TTSN:-300}" --s2t_n "${S2TN:-2423}" \
  --tts_cfg "${TTSCFG:-1.0,2.0}" --s2t_acfg "${S2TACFG:-1.0}" \
  --s2t_nfe "${S2TNFE:-128}" --nfe "${NFE:-32}" --gamma "${GAMMA:-0.0}" \
  --drop_text "${DROPTEXT:-0}" --skip_tts "${SKIPTTS:-0}" --skip_s2t "${SKIPS2T:-0}" \
  --out "$R/out/protocol_${RUN:-run}_p${PSEC:-3.0}_nfe${NFE:-32}_g${GAMMA:-0.0}_cfg${TTSCFG:-2.0}_dt${DROPTEXT:-0}.json"
