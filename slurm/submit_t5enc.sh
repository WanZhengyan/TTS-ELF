#!/bin/bash
#SBATCH --job-name=vf-t5enc
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/t5enc-%j.out
#SBATCH --error=logs/t5enc-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

echo "=== ELF init / T5 latent scaling probe ==="
echo "  job          : ${SLURM_JOB_ID:-local}"
echo "  node         : $(hostname)"
echo "  ROOT         : $ROOT"
echo "  started      : $(date -Is)"

source "$ROOT/.venv/bin/activate"
echo "  python       : $(which python)"
python -c "import torch;print('  torch        :',torch.__version__,'| cuda',torch.cuda.is_available())"

cd "$ROOT"
python probe_t5enc.py

echo "  finished     : $(date -Is)"
