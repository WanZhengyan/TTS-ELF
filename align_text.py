#!/usr/bin/env python
"""Real text-audio alignment from Whisper word timestamps.

Average upsampling assumes every character occupies the same number of frames.
On LJSpeech that is 2.66 frames each, which is wrong in detail: word durations
vary by an order of magnitude, and the model has been failing to learn exactly
the temporal structure that assumption throws away. Grad-TTS and Matcha-TTS --
the real LJSpeech baselines -- both use explicit per-unit durations, which is
what makes 24 hours enough.

This builds the same thing without MFA, using Whisper (already installed):

  1. transcribe each clip with word timestamps
  2. align Whisper's words to the ground-truth transcript by edit distance, so
     the conditioning text stays the text we evaluate against
  3. give each ground-truth word the matched word's (start, end); interpolate for
     words Whisper missed
  4. spread each word's characters over that word's own frames, and fill the gaps
     between words with the space character

Writes cache/align.pt: {clip_id: int16 array of per-frame character ids}.
Use it with `train.py --aligned_text`.
"""
import argparse, json, os, pickle, sys, time

import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))


def log(*a):
    print(*a, flush=True)


def norm_word(w):
    return "".join(c for c in w.lower() if c.isalnum())


def align_words(gt, hyp):
    """Levenshtein alignment of two word lists. Returns gt_index -> hyp_index or None."""
    n, m = len(gt), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (gt[i - 1] != hyp[j - 1]))
    out = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        if gt[i - 1] == hyp[j - 1] and d[i, j] == d[i - 1, j - 1]:
            out[i - 1] = j - 1; i -= 1; j -= 1
        elif d[i, j] == d[i - 1, j - 1] + 1:
            i -= 1; j -= 1                      # substitution: no reliable timing
        elif d[i, j] == d[i - 1, j] + 1:
            i -= 1                              # deletion
        else:
            j -= 1                              # insertion
    return out


def frames_for_words(gt_words, hyp_words, hyp_spans, T, frame_rate):
    """Per-ground-truth-word (start_frame, end_frame), interpolating the unmatched."""
    idx = align_words([norm_word(w) for w in gt_words],
                      [norm_word(w) for w in hyp_words])
    spans = [None] * len(gt_words)
    for gi, hi in enumerate(idx):
        if hi is None:
            continue
        s, e = hyp_spans[hi]
        if s is None or e is None or e <= s:
            continue
        spans[gi] = (max(0, int(s * frame_rate)), min(T, int(round(e * frame_rate))))
    # interpolate runs of unmatched words between two anchors
    anchors = [i for i, v in enumerate(spans) if v is not None]
    if not anchors:
        return None, 0.0
    if anchors[0] > 0:
        e0 = spans[anchors[0]][0]
        for k, i in enumerate(range(anchors[0])):
            a = int(e0 * k / anchors[0]); b = int(e0 * (k + 1) / anchors[0])
            spans[i] = (a, max(a + 1, b))
    if anchors[-1] < len(spans) - 1:
        s0 = spans[anchors[-1]][1]; n = len(spans) - 1 - anchors[-1]
        for k, i in enumerate(range(anchors[-1] + 1, len(spans))):
            a = s0 + int((T - s0) * k / n); b = s0 + int((T - s0) * (k + 1) / n)
            spans[i] = (a, max(a + 1, b))
    for lo, hi in zip(anchors, anchors[1:]):
        if hi == lo + 1:
            continue
        a0, b0 = spans[lo][1], spans[hi][0]
        n = hi - lo - 1
        for k, i in enumerate(range(lo + 1, hi)):
            a = a0 + int((b0 - a0) * k / n); b = a0 + int((b0 - a0) * (k + 1) / n)
            spans[i] = (a, max(a + 1, b))
    cov = len(anchors) / max(len(gt_words), 1)
    return spans, cov


def build_frame_text(gt_words, spans, T, stoi, pad=0):
    """Characters of each word spread over that word's frames; gaps become spaces."""
    space = stoi.get(" ", 1)
    out = np.full(T, space, dtype=np.int16)
    for w, sp in zip(gt_words, spans):
        if sp is None:
            continue
        a, b = sp
        a = max(0, min(T - 1, a)); b = max(a + 1, min(T, b))
        chars = [stoi.get(c, 1) for c in w.lower()]
        if not chars:
            continue
        n = b - a
        j = np.arange(n)
        out[a:b] = np.array(chars, dtype=np.int16)[np.minimum((j * len(chars)) // n,
                                                             len(chars) - 1)]
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(HERE, "cache"))
    ap.add_argument("--lj_root", default=os.path.join(HERE, "data", "LJSpeech-1.1"))
    ap.add_argument("--asr", default="openai/whisper-large-v3")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or os.path.join(args.cache, "align.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(args.cache, "index.pt"), "rb") as f:
        meta = pickle.load(f)
    recs = meta["records"]; stoi = meta["stoi"]; fr = meta["frame_rate"]
    if args.limit:
        recs = recs[:args.limit]
    log(f"{len(recs)} clips | frame_rate {fr} Hz | device {device}")

    id2text = {}
    with open(os.path.join(args.lj_root, "metadata.csv"), encoding="utf-8") as f:
        for line in f:
            p = line.strip().split("|")
            if len(p) >= 3:
                id2text[p[0]] = p[2]

    from transformers import pipeline
    log(f"loading {args.asr} ...")
    asr = pipeline("automatic-speech-recognition", model=args.asr,
                   device=0 if device.type == "cuda" else -1,
                   torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                   chunk_length_s=30, return_timestamps="word")

    align, covs, fpc = {}, [], []
    t0 = time.time(); fails = 0
    for i in range(0, len(recs), args.batch):
        chunk = recs[i:i + args.batch]
        waves = []
        for r in chunk:
            w, sr = torchaudio.load(os.path.join(args.lj_root, "wavs", r["id"] + ".wav"))
            w = w.mean(0) if w.shape[0] > 1 else w[0]
            waves.append({"sampling_rate": sr, "raw": w.numpy()})
        outs = asr(waves, generate_kwargs={"language": "english"})
        if isinstance(outs, dict):
            outs = [outs]
        for r, o in zip(chunk, outs):
            T = r["T"]
            gt = id2text.get(r["id"], "").split()
            chunks = o.get("chunks") or []
            hyp = [c["text"].strip() for c in chunks]
            spans_h = [tuple(c.get("timestamp") or (None, None)) for c in chunks]
            if not gt or not hyp:
                fails += 1; continue
            spans, cov = frames_for_words(gt, hyp, spans_h, T, fr)
            if spans is None:
                fails += 1; continue
            align[r["id"]] = build_frame_text(gt, spans, T, stoi)
            covs.append(cov)
            for w, sp in zip(gt, spans):
                if sp and len(w):
                    fpc.append((sp[1] - sp[0]) / len(w))
        if (i + args.batch) % (args.batch * 20) == 0:
            el = time.time() - t0
            log(f"  {min(i+args.batch,len(recs))}/{len(recs)}  {el:.0f}s  "
                f"({(i+args.batch)/max(el,1e-9):.1f} clip/s)  word coverage "
                f"{100*np.mean(covs):.1f}%")

    fpc = np.array(fpc)
    stats = {
        "n_aligned": len(align), "n_failed": fails,
        "word_coverage_mean": float(np.mean(covs)) if covs else 0.0,
        "frames_per_char": {"mean": float(fpc.mean()), "std": float(fpc.std()),
                            "p5": float(np.percentile(fpc, 5)),
                            "p50": float(np.percentile(fpc, 50)),
                            "p95": float(np.percentile(fpc, 95))},
        "uniform_baseline": meta["frames_per_char"],
    }
    with open(out_path, "wb") as f:
        pickle.dump({"align": align, "stats": stats}, f)
    log("")
    log(f"aligned {len(align)} clips, {fails} failed")
    log(f"word coverage (direct Whisper timing, rest interpolated): "
        f"{100*stats['word_coverage_mean']:.1f}%")
    log("")
    log("=== frames per character: real vs the uniform assumption ===")
    log(f"  uniform (what average upsampling assumed)  {meta['frames_per_char']:.2f}")
    log(f"  real  mean {fpc.mean():.2f}  std {fpc.std():.2f}  "
        f"p5 {np.percentile(fpc,5):.2f}  p50 {np.percentile(fpc,50):.2f}  "
        f"p95 {np.percentile(fpc,95):.2f}")
    log(f"  p95/p5 spread = {np.percentile(fpc,95)/max(np.percentile(fpc,5),1e-9):.1f}x")
    log("  -> that spread is exactly the information the uniform assumption discarded")
    log(f"\nwrote {out_path}")
    with open(os.path.join(HERE, "out", "align_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
