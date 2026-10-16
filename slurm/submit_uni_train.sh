#!/bin/bash
#SBATCH --job-name=vf-uni
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/uni-%j.out
#SBATCH --error=logs/uni-%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CACHE="${CACHE:-$ROOT/cache_libri_train_clean_100}"
OUT="${OUT:-$ROOT/runs/uni}"
STEPS="${STEPS:-40000}"
BATCH="${BATCH:-16}"
MAXLEN="${MAXLEN:-512}"
MAXTEXT="${MAXTEXT:-48}"
LAYOUT="${LAYOUT:-channel}"
PS2T="${PS2T:-0.5}"
LR="${LR:-1e-4}"
INITFRAC="${INITFRAC:-0.0}"
PMEAN="${PMEAN:--1.5}"
DECPMEAN="${DECPMEAN:-0.8}"
S2TW="${S2TW:-1.0}"
DECPROB="${DECPROB:-0.2}"
AUDIODROP="${AUDIODROP:-0.0}"
RESUME="${RESUME:-}"
NOCOND="${NOCOND:-0}"
GRADCKPT="${GRADCKPT:-0}"
EVALCACHE="${EVALCACHE:-}"
NGPU="${NGPU:-1}"   # >1 走 torchrun；--gres 必须在 sbatch 命令行上覆盖
WHISPER="${WHISPER:-0}"
SPK="${SPK:-0}"
TEXTDROP="${TEXTDROP:-0.0}"
PROMPTHI="${PROMPTHI:-0.3}"   # 评测协议给 3s prompt，在 4s clip 上是 75%
EVERY="${EVERY:-2000}"
EVALBATCHES="${EVALBATCHES:-4}"   # 4卡时每卡 batch 小，要调大才能保住评测样本数
TTSCACHE="${TTSCACHE:-}"     # 设了就走双数据源：TTS 行从这里取
S2TCACHES="${S2TCACHES:-}"   # 逗号分隔，S2T 行从这些 cache 取
TTSEVALCACHE="${TTSEVALCACHE:-}"   # TTS 监控评测用的 cache；空 = TTS 第一个 cache 的 in-set
KEEPEVERY="${KEEPEVERY:-0}"  # 每 N 步另存一份 ckpt_<step>.pt
LRSCHED="${LRSCHED:-constant}"   # constant | cosine（ELF-S2T：余弦退火到 0）
RESUME_FROM_LIVE="${RESUME_FROM_LIVE:-}"   # 另一个作业正在写的 ckpt：先复制到 RESUME 并验证能读

# Echo every derived value: a var set after the config string is built is
# silently ignored, so the log has to show what actually ran.
echo "=== unified TTS + S2T ==="
echo "  job     : ${SLURM_JOB_ID:-local}   node: $(hostname)"
echo "  CACHE   : $CACHE"
echo "  OUT     : $OUT"
echo "  STEPS   : $STEPS   BATCH: $BATCH   LR: $LR"
echo "  MAXLEN  : $MAXLEN   MAXTEXT: $MAXTEXT   P_S2T: $PS2T"
echo "  EVERY   : $EVERY   EVAL_BATCHES: $EVALBATCHES (约 $((EVALBATCHES*BATCH*NGPU)) clips)"
echo "  INITFRAC: $INITFRAC"
echo "  PMEAN   : $PMEAN"
echo "  LAYOUT  : $LAYOUT"
echo "  DECPMEAN: $DECPMEAN   S2TW: $S2TW   WHISPER: $WHISPER   SPK: $SPK"
echo "  TEXTDROP: $TEXTDROP   (TTS 行清零文本的概率，F5-TTS 用 0.2)"
echo "  PROMPT_HI: $PROMPTHI   (训练时 prompt 占 clip 的上限比例)"
echo "  DECPROB : $DECPROB   AUDIODROP: $AUDIODROP"
echo "  RESUME  : ${RESUME:-(none)}"
echo "  NOCOND  : $NOCOND   GRADCKPT: $GRADCKPT"
echo "  EVALCACHE: ${EVALCACHE:-(从 CACHE 切 val)}"
echo "  TTSCACHE : ${TTSCACHE:-(单数据源，用 CACHE)}"
echo "  S2TCACHES: ${S2TCACHES:-(单数据源，用 CACHE)}"
echo "  TTSEVALCACHE: ${TTSEVALCACHE:-(TTS 第一个 cache 的 in-set)}"
echo "  KEEPEVERY: $KEEPEVERY"
echo "  LRSCHED : $LRSCHED"
echo "  RESUME_FROM_LIVE: ${RESUME_FROM_LIVE:-(none)}"
echo "  NGPU    : $NGPU   (每卡 batch $BATCH -> 有效 batch $((BATCH*NGPU)))"
echo "  可见 GPU: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "  started : $(date -Is)"

# compute nodes have their own /tmp and one of them was full (Errno 28),
# so keep temp files on /fsx where there is room
export TMPDIR="$ROOT/tmp"
mkdir -p "$TMPDIR"
echo "  TMPDIR  : $TMPDIR"

source "$ROOT/.venv/bin/activate"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$ROOT"
if [ -n "$RESUME_FROM_LIVE" ]; then
  # the source is still being rewritten every eval by the job we are taking
  # over from; a copy taken mid-save would not load, so retry until one does
  ok=0; mkdir -p "$(dirname "$RESUME")"
  for k in 1 2 3 4 5 6; do
    cp "$RESUME_FROM_LIVE" "$RESUME" && python -c "import sys,torch; ck=torch.load(sys.argv[1],map_location='cpu',weights_only=False); print('  resume copy ok: step', ck['step'])" "$RESUME" && { ok=1; break; }
    echo "  resume copy attempt $k failed, retrying"; sleep 20
  done
  [ "$ok" = 1 ] || { echo "无法得到可读的 resume ckpt"; exit 1; }
fi
if [ "$NGPU" -gt 1 ]; then
  LAUNCH="torchrun --standalone --nproc_per_node=$NGPU"
else
  LAUNCH="python"
fi
$LAUNCH train_unified.py --cache "$CACHE" --out "$OUT" --steps "$STEPS" \
  --batch "$BATCH" --max_length "$MAXLEN" --max_text "$MAXTEXT" \
  --p_s2t "$PS2T" --lr "$LR" --eval_every "$EVERY" --eval_batches "$EVALBATCHES" --init_frac "$INITFRAC" --denoiser_p_mean "$PMEAN" --layout "$LAYOUT" --decoder_p_mean "$DECPMEAN" --s2t_weight "$S2TW" --whisper "$WHISPER" --spk "$SPK" --prompt_hi "$PROMPTHI" --decoder_prob "$DECPROB" --audio_drop_prob "$AUDIODROP" --text_drop_prob "$TEXTDROP" --resume "$RESUME" --no_cond "$NOCOND" --grad_ckpt "$GRADCKPT" --eval_cache "$EVALCACHE" \
  --tts_cache "$TTSCACHE" --s2t_caches "$S2TCACHES" --tts_eval_cache "$TTSEVALCACHE" --keep_every "$KEEPEVERY" --lr_schedule "$LRSCHED"

echo "  finished : $(date -Is)"
