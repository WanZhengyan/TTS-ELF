#!/bin/bash
#SBATCH --job-name=vf-prep960
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/prep960-%j.out
#SBATCH --error=logs/prep960-%j.err
# One job per cache, steps strictly in sequence. preprocess_whisper.py and
# preprocess_spk.py both read index.pt, add fields and write it back, so two of
# them running on the same cache at once would lose whichever wrote first.
# Combining them also keeps us under the 5-job submit limit.
set -uo pipefail
R="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CACHE="${CACHE:?CACHE 必须设置}"
SPLIT="${SPLIT:?SPLIT 必须设置（LibriSpeech 目录名，用来找 flac）}"
STEPS="${STEPS:-whisper,spk}"
echo "=== 960h 数据准备 ==="
echo "  job   : ${SLURM_JOB_ID:-local}  node: $(hostname)"
echo "  CACHE : $CACHE"
echo "  SPLIT : $SPLIT"
echo "  STEPS : $STEPS"
echo "  start : $(date -Is)"
export TMPDIR="$R/tmp"; mkdir -p "$TMPDIR"
source "$R/.venv/bin/activate"; cd "$R"
if [[ ",$STEPS," == *",whisper,"* ]]; then
  echo ""; echo "---- whisper-large-v3 特征 ----"
  python preprocess_whisper.py --cache "$CACHE" --split "$SPLIT" \
    --model openai/whisper-large-v3 || { echo "whisper 失败"; exit 1; }
fi
if [[ ",$STEPS," == *",spk,"* ]]; then
  echo ""; echo "---- UniSpeechSat x-vector（前 3 秒）----"
  python preprocess_spk.py --cache "$CACHE" --split "$SPLIT" \
    --prompt_sec 3.0 || { echo "x-vector 失败"; exit 1; }
fi
echo ""; echo "  done  : $(date -Is)"
