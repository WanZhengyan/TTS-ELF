#!/usr/bin/env python
"""Is the codec latent a bad flow target because it is temporally rough?

Every flow-matching TTS model that works -- Grad-TTS, Matcha-TTS, E2/F5-TTS,
ZipVoice -- generates MEL SPECTROGRAMS. We generate codec pre-quantization
latents instead. A neural codec is trained to make each frame maximally
informative for quantization, which decorrelates neighbouring frames; mel is
smooth by construction. If codec latents are much rougher in time, there is
simply less temporal structure for a flow to latch onto, and what remains is
high-frequency detail the model must get exactly right.

Three-way comparison, on the same clips where possible:

  T5 text latent   ELF's native target, where ELF demonstrably works
  mel spectrogram  what every working TTS flow model targets
  codec latent     what we target, and fail on

Metrics per representation, all on normalised features so scale cannot confound:

  lag-k cosine     cos(x_t, x_{t+k}) for k = 1,2,4,8,16. How much a frame tells
                   you about its neighbours.
  step size        ||x_{t+1} - x_t|| / ||x_t||. How far the sequence moves per
                   frame. A rough target forces the flow to resolve fine detail.
  spectral mass    share of temporal-FFT energy below 1/8 of Nyquist. High means
                   the sequence is dominated by slow components, i.e. learnable
                   structure; low means frame-to-frame noise.
"""
import argparse, json, os, pickle, sys

import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "third_party", "WavTokenizer"))
sys.path.insert(0, HERE)
from decoder.pretrained import WavTokenizer  # noqa: E402


def log(*a):
    print(*a, flush=True)


def analyse(seqs, name, lags=(1, 2, 4, 8, 16)):
    """seqs: list of (T, D) float arrays. Returns a dict of smoothness metrics."""
    cos = {k: [] for k in lags}
    step, lowf = [], []
    for x in seqs:
        if x.shape[0] < max(lags) + 2:
            continue
        # centre and unit-scale per dimension so representations are comparable
        x = (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-8)
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        u = x / n
        for k in lags:
            cos[k].append(float((u[:-k] * u[k:]).sum(1).mean()))
        step.append(float((np.linalg.norm(x[1:] - x[:-1], axis=1) / n[:-1, 0]).mean()))
        # temporal FFT: share of energy in the lowest eighth of the band
        F = np.abs(np.fft.rfft(x, axis=0)) ** 2
        tot = F[1:].sum()
        cut = max(1, F.shape[0] // 8)
        lowf.append(float(F[1:cut].sum() / (tot + 1e-12)))
    return {
        "name": name, "n_seqs": len(step),
        "cos": {f"lag{k}": float(np.mean(v)) for k, v in cos.items()},
        "step_rel": float(np.mean(step)),
        "lowfreq_share": float(np.mean(lowf)),
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(HERE, "cache"))
    ap.add_argument("--lj_root", default=os.path.join(HERE, "data", "LJSpeech-1.1"))
    ap.add_argument("--codec_cfg", default=os.path.join(
        HERE, "third_party", "WavTokenizer", "configs",
        "wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"))
    ap.add_argument("--codec_ckpt", default=os.path.join(
        HERE, "ckpt", "wavtokenizer_large_unify_600_24k.ckpt"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "smoothness.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(args.cache, "index.pt"), "rb") as f:
        meta = pickle.load(f)
    recs = meta["records"][:args.n]
    total = meta["total_frames"]
    lat = np.memmap(os.path.join(args.cache, "latents.f16"), dtype=np.float16,
                    mode="r", shape=(total, 512))
    id2text = {}
    with open(os.path.join(args.lj_root, "metadata.csv"), encoding="utf-8") as f:
        for line in f:
            p = line.strip().split("|")
            if len(p) >= 3:
                id2text[p[0]] = p[2]

    # ---- codec latents: straight from the cache
    codec = [np.asarray(lat[r["offset"]:r["offset"] + r["T"]], dtype=np.float32)
             for r in recs]
    log(f"codec latents: {len(codec)} seqs, dim {codec[0].shape[1]}, "
        f"mean T {np.mean([c.shape[0] for c in codec]):.0f}")

    # ---- mel: 80 bins, hop chosen to land near the codec's 40 Hz so the
    #      comparison is per-frame-rate fair rather than per-sample
    hop = 600                      # 24000 / 600 = 40 Hz, same grid as the codec
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000, n_fft=2048, hop_length=hop, n_mels=80, power=1.0).to(device)
    resample = torchaudio.transforms.Resample(22050, 24000).to(device)
    mel = []
    for r in recs:
        w, sr = torchaudio.load(os.path.join(args.lj_root, "wavs", r["id"] + ".wav"))
        w = resample(w.to(device))
        w = w.mean(0, keepdim=True) if w.shape[0] > 1 else w
        m = torch.log(melspec(w)[0] + 1e-5).transpose(0, 1)       # (T, 80)
        mel.append(m.float().cpu().numpy())
    log(f"mel: dim {mel[0].shape[1]}, mean T {np.mean([m.shape[0] for m in mel]):.0f}, "
        f"hop {hop} -> {24000/hop:.0f} Hz")

    # ---- T5 text latents: ELF's own target space
    t5 = []
    try:
        from transformers import T5EncoderModel, T5TokenizerFast
        tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        enc = T5EncoderModel.from_pretrained("google-t5/t5-small").to(device).eval()
        for r in recs:
            s = id2text.get(r["id"], "")
            if not s:
                continue
            ids = tok(s, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            h = enc(input_ids=ids).last_hidden_state[0]           # (N, 512)
            t5.append(h.float().cpu().numpy())
        log(f"T5 latents: dim {t5[0].shape[1]}, mean T {np.mean([x.shape[0] for x in t5]):.0f}")
    except Exception as e:
        log(f"T5 unavailable ({e}); continuing with two representations")

    res = [analyse(codec, "codec latent (ours)"),
           analyse(mel, "mel spectrogram (what works)")]
    if t5:
        res.append(analyse(t5, "T5 text latent (ELF's native)"))

    log("")
    log("=== temporal smoothness: how much structure is there to learn? ===")
    log(f"{'representation':<30} {'lag1':>7} {'lag2':>7} {'lag4':>7} {'lag8':>7} "
        f"{'lag16':>7} | {'step/|x|':>9} {'low-freq':>9}")
    log("-" * 96)
    for r in res:
        c = r["cos"]
        log(f"{r['name']:<30} {c['lag1']:>7.3f} {c['lag2']:>7.3f} {c['lag4']:>7.3f} "
            f"{c['lag8']:>7.3f} {c['lag16']:>7.3f} | {r['step_rel']:>9.3f} "
            f"{r['lowfreq_share']:>8.1%}")
    log("")
    log("lag-k cosine : higher = a frame predicts its neighbours, so there is")
    log("               temporal structure for a flow to learn.")
    log("step/|x|     : lower = the sequence moves gently; higher = the model must")
    log("               resolve large frame-to-frame jumps exactly.")
    log("low-freq     : share of temporal energy below 1/8 Nyquist. Higher = the")
    log("               sequence is dominated by slow, learnable components.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    log(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
