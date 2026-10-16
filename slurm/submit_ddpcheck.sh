#!/bin/bash
#SBATCH --job-name=vf-ddpchk
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/ddpchk-%j.out
#SBATCH --error=logs/ddpchk-%j.err
set -euo pipefail
R="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TMPDIR="$R/tmp"; mkdir -p "$TMPDIR"
source "$R/.venv/bin/activate"; cd "$R"
echo "=== DDP 梯度等价性：同一批固定数据，1 卡 32 条 vs 4 卡各 8 条 ==="
echo "  node: $(hostname)   可见 GPU: $(nvidia-smi -L | wc -l)"
echo ""
echo "---------- 1 rank, batch 32 ----------"
torchrun --standalone --nproc_per_node=1 ddp_check.py --batch 32 --total 32
echo ""
echo "---------- 4 ranks, batch 8 ----------"
torchrun --standalone --nproc_per_node=4 ddp_check.py --batch 8 --total 32
