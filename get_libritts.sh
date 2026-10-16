#!/bin/bash
# Mirror: openslr.elda.org measured 9.70 MB/s vs 0.36 MB/s for www.openslr.org
# (byte sizes verified identical, so resuming a partial file across mirrors is safe).
# Fetch LibriTTS 585h, the TTS training data of F5-TTS small (our TTS baseline)..
#
# One archive at a time, extracted and then deleted: downloading both first
# would peak at ~100 GB (50 GB of tarballs plus 50 GB extracted) against 148 GB
# free, and the mel cache still needs ~60 GB after this. Sequential keeps the
# peak near 79 GB.
#
# Resumable: curl -C - continues a partial file, and a split already extracted
# is skipped.
set -uo pipefail
ROOT="/fsx/homes/Qiang.Sun@mbzuai.ac.ae/voiceflow"
DATA="$ROOT/data/LibriTTS"
LOG="$ROOT/logs/get_libritts.log"
say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

free_gb(){ df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9'; }

say "开始；可用 $(free_gb) GB"
for split in train-clean-100 train-clean-360 train-other-500; do
  if [ -d "$DATA/$split" ] && [ "$(find "$DATA/$split" -name '*.wav' | head -1)" ]; then
    say "$split 已存在且含 flac，跳过"
    continue
  fi
  tar="$ROOT/data/libritts-${split}.tar.gz"   # 不能叫 ${split}.tar.gz：LibriSpeech 同名包还在，curl -C - 会把两者拼成坏包
  # 21.5 GB + 21.5 GB extracted for the smaller split, 28.5 + 28.5 for the
  # larger: require both plus a 15 GB cushion before starting
  case $split in
    train-clean-100) need=20 ;;
    train-clean-360) need=62 ;;
    *)               need=98 ;;
  esac
  avail=$(free_gb)
  if [ "$avail" -lt "$need" ]; then
    say "中止：$split 需要约 ${need} GB（下载+解压+余量），只有 ${avail} GB"
    exit 1
  fi
  say "下载 $split（可用 ${avail} GB，需要约 ${need} GB）"
  if ! curl -L -C - --retry 5 --retry-delay 10 --fail -o "$tar" \
       "https://openslr.elda.org/resources/60/${split}.tar.gz"; then
    say "下载 $split 失败"; exit 1
  fi
  say "下载完成 $(du -h "$tar" | cut -f1)，解压中"
  # the archive lays out LibriSpeech/<split>/..., so extract at data/
  if ! tar -xzf "$tar" -C "$ROOT/data"; then
    say "解压 $split 失败，保留压缩包以便重试"; exit 1
  fi
  rm -f "$tar"
  say "$split 解压完毕并删除压缩包；可用 $(free_gb) GB"
done

say "全部完成。各 split 的 flac 数："
for s in train-clean-100 train-clean-360 train-other-500; do
  [ -d "$DATA/$s" ] && say "  $s: $(find "$DATA/$s" -name '*.wav' | wc -l) wav, $(du -sh "$DATA/$s" | cut -f1)"
done
say "可用 $(free_gb) GB"
