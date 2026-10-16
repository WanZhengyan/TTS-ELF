#!/usr/bin/env python
"""Cache LJSpeech as log-mel, using Vocos's own feature extractor.

Why mel. Measured effective sequence complexity, T x (1 - lag1 cosine), which is
roughly how many independent things the flow must get right in one sample:

  T5 text latent   17   ELF works here
  mel @ 94 Hz      29   every TTS flow model works here
  mel @ 40 Hz      50
  codec latent    204   we fail here

Mel at Vocos's 93.75 Hz is 7x easier than the codec latent and lands in the same
regime as ELF's native target. The features come from vocos's own
MelSpectrogramFeatures so the pretrained vocoder matches exactly -- any mismatch
in n_fft, hop or padding silently degrades the audio.

Unlike the codec latent, log-mel is NOT centred (it sits near -5), so this also
measures latent_mean, which the codec path could leave at 0.
"""
import argparse, os, pickle, sys, time

import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))


def log(*a):
    print(*a, flush=True)


CHARS = " abcdefghijklmnopqrstuvwxyz'\".,;:!?-()"
PAD, UNK = 0, 1


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lj_root", default=os.path.join(HERE, "data", "LJSpeech-1.1"))
    ap.add_argument("--out", default=os.path.join(HERE, "cache_mel"))
    ap.add_argument("--min_s", type=float, default=1.0)
    ap.add_argument("--max_s", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from vocos.feature_extractors import MelSpectrogramFeatures
    # exactly Vocos's charactr/vocos-mel-24khz config
    fe = MelSpectrogramFeatures(sample_rate=24000, n_fft=1024, hop_length=256,
                                n_mels=100, padding="center").to(device).eval()
    D, FR = 100, 24000 / 256
    log(f"mel: n_mels {D}, hop 256 @ 24 kHz -> {FR:.2f} Hz (Vocos config)")

    stoi = {c: i + 2 for i, c in enumerate(CHARS)}
    vocab = len(CHARS) + 2

    rows = []
    with open(os.path.join(args.lj_root, "metadata.csv"), encoding="utf-8") as f:
        for line in f:
            p = line.strip().split("|")
            if len(p) >= 3:
                rows.append((p[0], p[2]))
    rows.sort()
    if args.limit:
        rows = rows[:args.limit]

    rs = torchaudio.transforms.Resample(22050, 24000).to(device)
    chunks, recs, total, skipped = [], [], 0, 0
    t0 = time.time()
    for i, (uid, text) in enumerate(rows):
        w, sr = torchaudio.load(os.path.join(args.lj_root, "wavs", uid + ".wav"))
        dur = w.shape[-1] / sr
        if not (args.min_s <= dur <= args.max_s):
            skipped += 1
            continue
        x = rs(w.to(device))
        x = x.mean(0, keepdim=True) if x.shape[0] > 1 else x
        m = fe(x)[0].transpose(0, 1).float()          # (T, 100) log-mel
        T = m.shape[0]
        chunks.append(m.half().cpu().numpy())
        recs.append({"id": uid, "offset": total, "T": T,
                     "chars": np.array([stoi.get(c, UNK) for c in text.lower().strip()],
                                       dtype=np.int16),
                     "code": np.zeros(T, dtype=np.int16),   # unused on this path
                     "dur": dur})
        total += T
        if (i + 1) % 2000 == 0:
            el = time.time() - t0
            log(f"  {i+1}/{len(rows)}  kept {len(recs)}  frames {total}  "
                f"{el:.0f}s ({(i+1)/el:.1f} clip/s)")

    path = os.path.join(args.out, "latents.f16")
    mm = np.memmap(path, dtype=np.float16, mode="w+", shape=(total, D))
    at = 0
    for a in chunks:
        mm[at:at + a.shape[0]] = a; at += a.shape[0]
    mm.flush(); del mm
    log(f"wrote {path} ({total*D*2/1e9:.2f} GB)")

    mm = np.memmap(path, dtype=np.float16, mode="r", shape=(total, D))
    samp = mm[np.random.default_rng(0).choice(total, min(400_000, total),
                                              replace=False)].astype(np.float32)
    lm, ls = float(samp.mean()), float(samp.std())
    Ts = np.array([r["T"] for r in recs]); Ns = np.array([len(r["chars"]) for r in recs])
    meta = {"records": recs, "total_frames": total, "dim": D,
            "char_vocab_size": vocab, "stoi": stoi,
            "audio_vocab_size": 2,              # no discrete target on this path
            "frame_rate": FR,
            "latent_mean": lm, "latent_std": ls,
            "frames": {"mean": float(Ts.mean()), "p95": float(np.percentile(Ts, 95)),
                       "max": int(Ts.max())},
            "chars_per_clip": {"mean": float(Ns.mean()), "max": int(Ns.max())},
            "frames_per_char": float((Ts / np.maximum(Ns, 1)).mean()),
            "target": "mel"}
    with open(os.path.join(args.out, "index.pt"), "wb") as f:
        pickle.dump(meta, f)

    log("")
    log(f"kept {len(recs)} clips, skipped {skipped}, {total} frames")
    log("=== log-mel is NOT centred, so latent_mean matters here ===")
    log(f"  latent_mean  {lm:+.5f}   <- the codec path could leave this at 0")
    log(f"  latent_std   {ls:.5f}")
    log(f"  frames/clip  mean {Ts.mean():.0f}  p95 {np.percentile(Ts,95):.0f}  max {Ts.max()}")
    log(f"  frames/char  mean {meta['frames_per_char']:.2f}")
    log(f"  suggested max_length  {int(np.percentile(Ts,99)//16*16+16)}")


if __name__ == "__main__":
    main()
