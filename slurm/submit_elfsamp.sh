#!/bin/bash
#SBATCH --job-name=vf-elfsamp
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=logs/elfsamp-%j.out
#SBATCH --error=logs/elfsamp-%j.err

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
python probe_elf_sample.py

echo "  finished     : $(date -Is)"
