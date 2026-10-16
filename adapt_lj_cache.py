#!/usr/bin/env python
"""Make the existing LJSpeech mel cache readable by the unified trainer.

This exists for one control experiment. The configuration that produced
intelligible speech was: LJSpeech (ONE speaker), pure TTS, from scratch,
text_encoder_dim 100, decode head off, 100k steps. The unified runs are:
LibriSpeech (251 speakers), ELF-B init, text_encoder_dim 512, decode head on,
and so far 7k steps -- and they do not produce speech (UTMOS 1.93 against a
3.67 ceiling with a 30% voice prompt, 1.35 without).

Five candidate causes have been ruled out by experiment (task mixing, the
sequence layout, the init scale of the new input columns, the training t
distribution, SDE sampling, and an output gain). What is left is the DATA
(multi-speaker, where text -> speech is one-to-many in speaker identity and the
L2 optimum is a speaker average) and the CODE/INIT. Running the unified
trainer on LJSpeech separates them: if it works, the data is the problem; if it
does not, the unified code or the ELF init is.

The mel latents are reused as they are -- only the index gains the fields the
unified pipeline needs: t5 token ids per clip, a speaker id (0 for everyone,
since there is one reader), and the frames-per-token rate.
"""
import os, pickle, shutil, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def log(*a):
    print(*a, flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "cache_mel"))
    ap.add_argument("--dst", default=os.path.join(HERE, "cache_lj_uni"))
    ap.add_argument("--lj_root", default=os.path.join(HERE, "data", "LJSpeech-1.1"))
    a = ap.parse_args()

    os.makedirs(a.dst, exist_ok=True)
    with open(os.path.join(a.src, "index.pt"), "rb") as f:
        meta = pickle.load(f)
    log(f"source: {len(meta['records'])} clips, {meta['total_frames']} frames, "
        f"dim {meta['dim']}")

    id2text = {}
    with open(os.path.join(a.lj_root, "metadata.csv"), encoding="utf-8") as f:
        for line in f:
            p = line.strip().split("|")
            if len(p) >= 3:
                id2text[p[0]] = p[2]

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("t5-small")

    Ks = []
    for r in meta["records"]:
        txt = id2text.get(r["id"], "").lower().strip()
        r["t5"] = np.asarray(tk(txt).input_ids, dtype=np.int32)
        r["spk"] = 0                      # LJSpeech is a single reader
        Ks.append(len(r["t5"]))
    Ks = np.array(Ks)
    Ts = np.array([r["T"] for r in meta["records"]])

    meta["speakers"] = [0]
    meta["n_speakers"] = 1
    meta["t5_vocab_size"] = 32100
    meta["t5_tokens"] = {"mean": float(Ks.mean()),
                         "p99": float(np.percentile(Ks, 99)),
                         "max": int(Ks.max())}
    meta["frames_per_t5"] = float((Ts / np.maximum(Ks, 1)).mean())
    meta["dataset"] = "ljspeech"
    meta["split"] = "LJSpeech-1.1"

    with open(os.path.join(a.dst, "index.pt"), "wb") as f:
        pickle.dump(meta, f)

    # reuse the latents in place rather than copying 1.58 GB
    # absolute: a relative target resolves against the LINK's directory, so
    # --src cache_x would point cache_y/latents.f16 at cache_y/cache_x/...
    src_lat = os.path.abspath(os.path.join(a.src, "latents.f16"))
    dst_lat = os.path.join(a.dst, "latents.f16")
    if not os.path.exists(dst_lat):
        os.symlink(src_lat, dst_lat)
        log(f"symlinked {dst_lat} -> {src_lat}")

    log("")
    log(f"wrote {a.dst}")
    log(f"  speakers        1  (vs 251 on LibriSpeech train-clean-100)")
    log(f"  t5 tokens       mean {Ks.mean():.1f}  p99 "
        f"{np.percentile(Ks, 99):.0f}  max {Ks.max()}")
    log(f"  frames per t5   {meta['frames_per_t5']:.1f}  "
        f"(LibriSpeech: 26.8)")
    log(f"  frames/clip     mean {Ts.mean():.0f}  p95 "
        f"{np.percentile(Ts, 95):.0f}  max {Ts.max()}")
    log(f"  latent_mean {meta['latent_mean']:+.5f}  "
        f"latent_std {meta['latent_std']:.5f}")


if __name__ == "__main__":
    main()
