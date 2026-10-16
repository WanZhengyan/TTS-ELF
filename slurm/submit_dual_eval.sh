#!/bin/bash
#SBATCH --job-name=vf-deval
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/deval-%j.out
#SBATCH --error=logs/deval-%j.err
# Both comparisons for one checkpoint of the two-source run:
#   TTS  eval_lj.py        F5-TTS small, Table 9 LJSpeech column (in-set 1000)
#   S2T  eval_s2t_full.py  ELF-S2T, all 2,620 test-clean utterances
set -uo pipefail
R="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CKPT="${CKPT:?CKPT 必须设置}"
TAG="${TAG:-$(basename "$(dirname "$CKPT")")_$(basename "$CKPT" .pt)}"
TTSCFG="${TTSCFG:-1.0,2.0}"
S2TACFG="${S2TACFG:-1.0,2.0}"
S2TNFE="${S2TNFE:-128}"
DOTTS="${DOTTS:-1}"
DOS2T="${DOS2T:-1}"
echo "=== 双数据源评测 ==="
echo "  job    : ${SLURM_JOB_ID:-local}  node: $(hostname)"
echo "  CKPT   : $CKPT"
echo "  TAG    : $TAG"
echo "  TTS    : $DOTTS  CFG $TTSCFG   (F5 用 2.0)"
echo "  S2T    : $DOS2T  ACFG $S2TACFG  NFE $S2TNFE   (ELF-S2T 用 w=2.0, K=128)"
echo "  start  : $(date -Is)"
export TMPDIR="$R/tmp"; mkdir -p "$TMPDIR" "$R/out/dual"
source "$R/.venv/bin/activate"; cd "$R"
rc=0
if [ "$DOTTS" = 1 ]; then
  echo ""; echo "---- TTS: LJSpeech in-set (F5 Table 9) ----"
  python eval_lj.py --ckpt "$CKPT" --cfg "$TTSCFG" \
    --out "out/dual/${TAG}_lj.json" || { echo "TTS 评测失败"; rc=1; }
fi
if [ "$DOS2T" = 1 ]; then
  echo ""; echo "---- S2T: full test-clean (ELF-S2T) ----"
  python eval_s2t_full.py --ckpt "$CKPT" --acfg "$S2TACFG" --nfe "$S2TNFE" \
    --out "out/dual/${TAG}_s2t.json" || { echo "S2T 评测失败"; rc=1; }
fi
echo ""; echo "  done   : $(date -Is)  rc=$rc"
exit $rc
