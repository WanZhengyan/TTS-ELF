#!/bin/bash
# Fetch the rest of LibriSpeech 960h so our training data matches the papers'.
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
DATA="$ROOT/data/LibriSpeech"
LOG="$ROOT/logs/get_libri960.log"
say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

free_gb(){ df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9'; }

say "开始；可用 $(free_gb) GB"
for split in train-clean-360 train-other-500; do
  if [ -d "$DATA/$split" ] && [ "$(find "$DATA/$split" -name '*.flac' | head -1)" ]; then
    say "$split 已存在且含 flac，跳过"
    continue
  fi
  tar="$ROOT/data/${split}.tar.gz"
  # 21.5 GB + 21.5 GB extracted for the smaller split, 28.5 + 28.5 for the
  # larger: require both plus a 15 GB cushion before starting
  case $split in
    train-clean-360) need=58 ;;
    *)               need=72 ;;
  esac
  avail=$(free_gb)
  if [ "$avail" -lt "$need" ]; then
    say "中止：$split 需要约 ${need} GB（下载+解压+余量），只有 ${avail} GB"
    exit 1
  fi
  say "下载 $split（可用 ${avail} GB，需要约 ${need} GB）"
  if ! curl -L -C - --retry 5 --retry-delay 10 --fail -o "$tar" \
       "https://www.openslr.org/resources/12/${split}.tar.gz"; then
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
for s in train-clean-100 train-clean-360 train-other-500 dev-clean test-clean; do
  [ -d "$DATA/$s" ] && say "  $s: $(find "$DATA/$s" -name '*.flac' | wc -l) flac, $(du -sh "$DATA/$s" | cut -f1)"
done
say "可用 $(free_gb) GB"
