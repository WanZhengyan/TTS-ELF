#!/usr/bin/env python
"""Cache LJSpeech as ELF-TTS training tensors.

For each clip:
  latent   (T, 512) fp16   WavTokenizer pre-quantization latent  -> the flow target x0
  code     (T,)     int16  single-codebook id in [0, 4096)        -> the CE target
  chars    (N,)     int16  character ids of the normalized text

Average upsampling is applied at batch time, not here, so the ratio stays exact
for whatever T a clip has.

Writes to voiceflow/cache/:
  latents.f16     one flat memmap, (sum T, 512)
  index.pt        per-clip offsets, codes, chars, vocab, stats
"""
import argparse, os, sys, time, pickle

import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "third_party", "WavTokenizer"))
from decoder.pretrained import WavTokenizer  # noqa: E402


def log(*a):
    print(*a, flush=True)


# Characters kept, in a fixed order so the vocab is reproducible.
# 0 is reserved as <pad>/filler, 1 as <unk>.
CHARS = " abcdefghijklmnopqrstuvwxyz'\".,;:!?-()"
PAD, UNK = 0, 1


def build_vocab():
    stoi = {c: i + 2 for i, c in enumerate(CHARS)}
    return stoi, len(CHARS) + 2


def encode_text(text, stoi):
    t = text.lower().strip()
    return np.array([stoi.get(c, UNK) for c in t], dtype=np.int16)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lj_root", default=os.path.join(HERE, "data", "LJSpeech-1.1"))
    ap.add_argument("--config", default=os.path.join(
        HERE, "third_party", "WavTokenizer", "configs",
        "wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"))
    ap.add_argument("--ckpt", default=os.path.join(HERE, "ckpt", "wavtokenizer_large_unify_600_24k.ckpt"))
    ap.add_argument("--out", default=os.path.join(HERE, "cache"))
    ap.add_argument("--max_s", type=float, default=10.0)
    ap.add_argument("--min_s", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all clips")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device {device}")

    model = WavTokenizer.from_pretrained0802(args.config, args.ckpt).to(device).eval()
    fe = model.feature_extractor.encodec
    vq = fe.quantizer.vq.layers[0]
    assert isinstance(vq.project_in, torch.nn.Identity)
    log("codec loaded")

    stoi, vocab_size = build_vocab()
    log(f"char vocab {vocab_size} (0=pad/filler, 1=unk)")

    # -- read metadata
    rows = []
    with open(os.path.join(args.lj_root, "metadata.csv"), encoding="utf-8") as f:
        for line in f:
            p = line.strip().split("|")
            if len(p) >= 3:
                rows.append((p[0], p[2]))
    rows.sort()
    if args.limit:
        rows = rows[:args.limit]
    log(f"{len(rows)} clips in metadata")

    resample = torchaudio.transforms.Resample(22050, 24000).to(device)

    # -- pass: encode, accumulate
    lat_chunks, recs, total = [], [], 0
    t0 = time.time()
    skipped = 0
    for i, (uid, text) in enumerate(rows):
        wav, sr = torchaudio.load(os.path.join(args.lj_root, "wavs", uid + ".wav"))
        dur = wav.shape[-1] / sr
        if not (args.min_s <= dur <= args.max_s):
            skipped += 1
            continue
        w = resample(wav.to(device))
        if w.shape[0] > 1:
            w = w.mean(0, keepdim=True)
        emb = fe.encoder(w.unsqueeze(1))                 # (1, 512, T)
        code = vq.encode(emb)[0]                         # (T,)
        z = emb[0].transpose(0, 1)                       # (T, 512)
        T = z.shape[0]
        assert code.shape[0] == T, f"{uid}: code {code.shape} vs latent {T}"

        lat_chunks.append(z.half().cpu().numpy())
        recs.append({
            "id": uid, "offset": total, "T": T,
            "code": code.to(torch.int16).cpu().numpy(),
            "chars": encode_text(text, stoi),
            "dur": dur,
        })
        total += T
        if (i + 1) % 1000 == 0:
            el = time.time() - t0
            log(f"  {i+1}/{len(rows)}  kept {len(recs)}  frames {total}  "
                f"{el:.0f}s  ({(i+1)/el:.1f} clip/s)")

    log(f"kept {len(recs)} clips, skipped {skipped}, {total} frames total")

    # -- write latents memmap
    path = os.path.join(args.out, "latents.f16")
    mm = np.memmap(path, dtype=np.float16, mode="w+", shape=(total, 512))
    at = 0
    for a in lat_chunks:
        mm[at:at + a.shape[0]] = a
        at += a.shape[0]
    mm.flush()
    del mm
    log(f"wrote {path}  ({total*512*2/1e9:.2f} GB)")

    # -- stats on the cached latents (confirms the Test 0 number on the full set)
    mm = np.memmap(path, dtype=np.float16, mode="r", shape=(total, 512))
    samp = mm[np.random.default_rng(0).choice(total, min(400_000, total), replace=False)].astype(np.float32)
    latent_mean, latent_std = float(samp.mean()), float(samp.std())
    log(f"full-set latent mean {latent_mean:+.5f}  std {latent_std:.5f}")

    Ts = np.array([r["T"] for r in recs])
    Ns = np.array([len(r["chars"]) for r in recs])
    meta = {
        "records": recs, "total_frames": total, "dim": 512,
        "char_vocab_size": vocab_size, "stoi": stoi,
        "audio_vocab_size": 4096, "frame_rate": 40.0,
        "latent_mean": latent_mean, "latent_std": latent_std,
        "frames": {"mean": float(Ts.mean()), "p95": float(np.percentile(Ts, 95)), "max": int(Ts.max())},
        "chars_per_clip": {"mean": float(Ns.mean()), "max": int(Ns.max())},
        "frames_per_char": float((Ts / np.maximum(Ns, 1)).mean()),
    }
    with open(os.path.join(args.out, "index.pt"), "wb") as f:
        pickle.dump(meta, f)
    log(f"wrote index.pt")
    log(f"frames/clip  mean {Ts.mean():.0f}  p95 {np.percentile(Ts,95):.0f}  max {Ts.max()}")
    log(f"chars/clip   mean {Ns.mean():.0f}  max {Ns.max()}")
    log(f"frames/char  mean {meta['frames_per_char']:.2f}   <- the average-upsampling ratio")


if __name__ == "__main__":
    main()
