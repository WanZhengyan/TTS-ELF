#!/bin/bash
#SBATCH --job-name=vf-metric
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/metric-%j.out
#SBATCH --error=logs/metric-%j.err
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N="${N:-100}"; EXTRA="${EXTRA:-}"; OUT="${OUT:-$R/out/metrics.json}"
echo "N=$N EXTRA=$EXTRA OUT=$OUT  node $(hostname)"
source "$R/.venv/bin/activate"; cd "$R"
python eval_metrics.py --n "$N" --out "$OUT" $EXTRA
