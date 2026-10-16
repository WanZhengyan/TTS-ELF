#!/usr/bin/env python
"""Fork a cache so one set of mels can serve several frozen encoders.

`preprocess_whisper.py` writes its features into the cache directory and
rewrites `index.pt` in place, overwriting `whisper_dim`, `whisper_total` and
friends. Running it twice with different encoders therefore makes the first
encoder's features unreachable -- the file survives but nothing knows its
shape -- and would also change the meta underneath a training run that is
using it.

So each encoder gets its own cache directory: `latents.f16` is symlinked (no
copy of several GB) and `index.pt` is a private copy with the whisper fields
stripped, because the new directory has no feature file yet and LibriCache
would try to memmap one. The precompute writes them back.
"""
import os
import pickle
import sys


def fork(src, dst):
    if not os.path.isdir(src):
        sys.exit(f"源 cache 不存在: {src}")
    if os.path.exists(dst):
        print(f"目标已存在，不动: {dst}")
        return
    os.makedirs(dst)
    os.symlink(os.path.join(os.path.abspath(src), "latents.f16"),
               os.path.join(dst, "latents.f16"))
    with open(os.path.join(src, "index.pt"), "rb") as f:
        meta = pickle.load(f)
    dropped = [k for k in ("whisper_file", "whisper_dim", "whisper_model",
                           "whisper_rate", "whisper_total") if k in meta]
    for k in dropped:
        meta.pop(k)
    for r in meta["records"]:
        r.pop("w_offset", None)
        r.pop("w_T", None)
    with open(os.path.join(dst, "index.pt"), "wb") as f:
        pickle.dump(meta, f)
    print(f"派生 {src} -> {dst}   {len(meta['records'])} 条记录, "
          f"latents 软链, 剥除字段 {dropped or '(无)'}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("用法: fork_cache.py <源 cache> <目标 cache>")
    fork(sys.argv[1], sys.argv[2])
