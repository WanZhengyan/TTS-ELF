#!/bin/bash
#SBATCH --job-name=vf-melEval
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/melEval-%j.out
#SBATCH --error=logs/melEval-%j.err
set -euo pipefail
R="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$R/.venv/bin/activate"; cd "$R"
echo "### does it work? diversity / lag1 / WER / UTMOS / mel ceiling"
python sample_mel.py --n 50 --n_wav 6 --weights ema
echo
echo "### raw weights too, in case EMA lags"
python sample_mel.py --n 30 --n_wav 0 --weights model \
  --page "$R/listen_mel_raw.html" --out "$R/out/metrics_mel_raw.json"
echo
echo "### error bar on the ratio metric"
python eval_noise.py --reps 12 --weights raw
