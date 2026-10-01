#!/usr/bin/env python
"""Test 0 — the no-training codec probe for ELF-TTS.

Answers the precondition behind ELF's design: are WavTokenizer's codebook cells
large relative to the error a flow model will land with at t=1? Nothing here
trains; it measures the geometry of the latent space the flow would live in.

Outputs (voiceflow/out/):
  test0.json   scalar findings, incl. the latent_std every later config needs
  test0.npz    raw arrays for plotting
  wav/         clean resynthesis + resynthesis at several perturbation levels

Run via slurm/submit_probe.sh — needs a GPU for the encoder passes.
"""
import argparse, json, os, sys, time

import numpy as np
import torch
import torchaudio

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "WavTokenizer")
sys.path.insert(0, REPO)
from decoder.pretrained import WavTokenizer  # noqa: E402


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- data
def load_clips(lj_root, n, seed, min_s, max_s, src_sr, tgt_sr, device):
    """Read n LJSpeech clips, resample to tgt_sr, keep those inside the duration window."""
    meta = os.path.join(lj_root, "metadata.csv")
    ids = []
    with open(meta, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 3:
                ids.append((parts[0], parts[2]))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    resample = torchaudio.transforms.Resample(src_sr, tgt_sr).to(device)
    out, skipped = [], 0
    for uid, text in ids:
        if len(out) >= n:
            break
        path = os.path.join(lj_root, "wavs", uid + ".wav")
        wav, sr = torchaudio.load(path)
        assert sr == src_sr, f"{uid}: expected {src_sr} Hz, got {sr}"
        dur = wav.shape[-1] / sr
        if not (min_s <= dur <= max_s):
            skipped += 1
            continue
        wav = resample(wav.to(device))
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        out.append({"id": uid, "text": text, "wav": wav, "dur": dur})
    log(f"  loaded {len(out)} clips, skipped {skipped} outside [{min_s}, {max_s}] s")
    return out


# ----------------------------------------------------------------------------- probe
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--lj_root", default=os.path.join(here, "data", "LJSpeech-1.1"))
    ap.add_argument("--config", default=os.path.join(
        REPO, "configs", "wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"))
    ap.add_argument("--ckpt", default=os.path.join(here, "ckpt", "wavtokenizer_large_unify_600_24k.ckpt"))
    ap.add_argument("--out_dir", default=os.path.join(here, "out"))
    ap.add_argument("--n_clips", type=int, default=500)
    ap.add_argument("--n_wav_dump", type=int, default=8)
    ap.add_argument("--max_frames_per_clip", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_s", type=float, default=1.0)
    ap.add_argument("--max_s", type=float, default=10.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "wav"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    log(f"device: {device}")
    if device.type != "cuda":
        log("  WARNING: no GPU — this will be slow")

    # -- model
    t0 = time.time()
    model = WavTokenizer.from_pretrained0802(args.config, args.ckpt).to(device).eval()
    fe = model.feature_extractor.encodec
    vq = fe.quantizer.vq.layers[0]
    assert isinstance(vq.project_in, torch.nn.Identity), \
        "project_in is not Identity — margins would need measuring in the projected space"
    codebook = vq._codebook.embed.detach().to(device).float()      # (K, D)
    K, D = codebook.shape
    log(f"model loaded in {time.time()-t0:.1f}s | codebook {K} x {D}")

    # frame rate implied by the encoder's total stride
    hop = int(np.prod(fe.encoder.ratios)) if hasattr(fe.encoder, "ratios") else 600
    frame_rate = 24000 / hop
    log(f"hop {hop} samples -> {frame_rate:.1f} Hz")

    # -- data
    clips = load_clips(args.lj_root, args.n_clips, args.seed,
                       args.min_s, args.max_s, 22050, 24000, device)

    # -- encode: collect pre-quantization latents
    lat, per_clip_frames, durs = [], [], []
    rng = np.random.default_rng(args.seed + 1)
    for c in clips:
        emb = fe.encoder(c["wav"].unsqueeze(1))          # (1, D, T)
        z = emb[0].transpose(0, 1).float()               # (T, D)
        per_clip_frames.append(z.shape[0])
        durs.append(c["dur"])
        if z.shape[0] > args.max_frames_per_clip:
            idx = rng.choice(z.shape[0], args.max_frames_per_clip, replace=False)
            z = z[np.sort(idx)]
        lat.append(z)
    Z = torch.cat(lat, 0)                                # (N, D)
    N = Z.shape[0]
    log(f"collected {N} frames from {len(clips)} clips")

    # ---------------------------------------------------------------- 1. latent stats
    latent_mean = Z.mean().item()
    latent_std = Z.std().item()
    per_dim_std = Z.std(0)
    frame_norm = Z.norm(dim=1)
    log("")
    log("=== 1. latent statistics (pre-quantization, the flow target space) ===")
    log(f"  global mean          {latent_mean:+.5f}")
    log(f"  global std           {latent_std:.5f}      <-- latent_std for the ELF config")
    log(f"  per-dim std  min/med/max   {per_dim_std.min():.4f} / "
        f"{per_dim_std.median():.4f} / {per_dim_std.max():.4f}")
    log(f"  ||z|| per frame  mean {frame_norm.mean():.3f}  std {frame_norm.std():.3f}")
    log(f"  ELF's default latent_std is 0.2 (a T5-small number) -> "
        f"ratio {latent_std/0.2:.2f}x")

    # ---------------------------------------------------------------- 2. codebook margins
    log("")
    log("=== 2. codebook geometry ===")
    cb_sq = codebook.pow(2).sum(1)                       # (K,)
    chunk = 8192
    d1 = torch.empty(N, device=device)
    d2 = torch.empty(N, device=device)
    code0 = torch.empty(N, dtype=torch.long, device=device)
    for i in range(0, N, chunk):
        x = Z[i:i + chunk]
        sq = (x.pow(2).sum(1, keepdim=True) - 2 * x @ codebook.t() + cb_sq[None, :])
        top2 = torch.topk(sq.clamp_min(0), 2, dim=1, largest=False)
        dd = top2.values.sqrt()
        d1[i:i + chunk], d2[i:i + chunk] = dd[:, 0], dd[:, 1]
        code0[i:i + chunk] = top2.indices[:, 0]

    rel_margin = ((d2 - d1) / d1.clamp_min(1e-8))
    quant_err = d1 / frame_norm.clamp_min(1e-8)
    used = torch.unique(code0).numel()

    log(f"  d1 (to nearest code)   mean {d1.mean():.4f}  median {d1.median():.4f}")
    log(f"  d2 (to 2nd nearest)    mean {d2.mean():.4f}")
    log(f"  relative margin (d2-d1)/d1:")
    for q in (5, 25, 50, 75, 95):
        log(f"      p{q:<3} {torch.quantile(rel_margin, q/100).item():.4f}")
    for thr in (0.05, 0.10, 0.25):
        frac = (rel_margin < thr).float().mean().item()
        log(f"  frames with margin < {thr:.2f}:  {100*frac:.2f}%")
    log(f"  relative quantization error ||z-c|| / ||z||:  "
        f"mean {quant_err.mean():.4f}  p95 {torch.quantile(quant_err, 0.95).item():.4f}")
    log(f"  codebook usage: {used}/{K} entries ({100*used/K:.1f}%) on this sample")

    # ---------------------------------------------------------------- 3. noise tolerance
    log("")
    log("=== 3. path A — nearest-codebook robustness to perturbation ===")
    log("  sigma is in units of latent_std; survival = fraction of frames keeping their code")
    sigmas = [0.0, 0.01, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00]
    survival = []
    for s in sigmas:
        if s == 0.0:
            survival.append(1.0)
            log(f"  sigma {s:<5.3f}  survival 100.00%")
            continue
        keep = 0
        for i in range(0, N, chunk):
            x = Z[i:i + chunk]
            xn = x + torch.randn_like(x) * (s * latent_std)
            sq = (xn.pow(2).sum(1, keepdim=True) - 2 * xn @ codebook.t() + cb_sq[None, :])
            keep += (sq.argmin(1) == code0[i:i + chunk]).sum().item()
        f = keep / N
        survival.append(f)
        log(f"  sigma {s:<5.3f}  survival {100*f:6.2f}%")

    def sigma_at(target):
        """Interpolate the sigma where survival crosses `target`."""
        for i in range(1, len(sigmas)):
            if survival[i] < target <= survival[i - 1]:
                lo, hi = survival[i], survival[i - 1]
                w = (target - lo) / (hi - lo) if hi > lo else 0.0
                return sigmas[i] + w * (sigmas[i - 1] - sigmas[i])
        return float("nan")

    s95, s99 = sigma_at(0.95), sigma_at(0.99)
    log("")
    log(f"  sigma_A @ 99% survival = {s99:.4f} x latent_std")
    log(f"  sigma_A @ 95% survival = {s95:.4f} x latent_std   <-- the error budget to beat")

    # ---------------------------------------------------------------- 4. resynthesis
    log("")
    log("=== 4. resynthesis — ceiling audio and perturbed audio for later WER/UTMOS ===")
    dump_sigmas = [0.0, 0.05, 0.10, 0.20]
    manifest = []
    bw = torch.tensor([0], device=device)
    for c in clips[:args.n_wav_dump]:
        emb = fe.encoder(c["wav"].unsqueeze(1))
        # ground truth passthrough, for reference
        torchaudio.save(os.path.join(args.out_dir, "wav", f"{c['id']}_orig.wav"),
                        c["wav"].cpu(), 24000, encoding="PCM_S", bits_per_sample=16)
        for s in dump_sigmas:
            e = emb if s == 0.0 else emb + torch.randn_like(emb) * (s * latent_std)
            idx = vq.encode(e)                              # (1, T) code ids
            q = vq.decode(idx)                              # (1, D, T) quantized
            audio = model.decode(q, bandwidth_id=bw)
            tag = "clean" if s == 0.0 else f"sig{s:.2f}"
            fn = f"{c['id']}_{tag}.wav"
            torchaudio.save(os.path.join(args.out_dir, "wav", fn),
                            audio.cpu(), 24000, encoding="PCM_S", bits_per_sample=16)
        manifest.append({"id": c["id"], "text": c["text"], "dur": c["dur"]})
    log(f"  wrote {args.n_wav_dump} x {len(dump_sigmas)+1} wavs to {args.out_dir}/wav")

    # ---------------------------------------------------------------- save
    frames = np.array(per_clip_frames)
    res = {
        "codec": "WavTokenizer-large-unify-40token (wavtokenizer_large_unify_600_24k)",
        "codebook": {"size": K, "dim": D, "used_on_sample": used},
        "frame_rate_hz": frame_rate,
        "n_clips": len(clips), "n_frames_sampled": N,
        "latent": {
            "mean": latent_mean, "std": latent_std,
            "per_dim_std_min": per_dim_std.min().item(),
            "per_dim_std_median": per_dim_std.median().item(),
            "per_dim_std_max": per_dim_std.max().item(),
            "frame_norm_mean": frame_norm.mean().item(),
            "elf_default_latent_std": 0.2,
            "ratio_to_elf_default": latent_std / 0.2,
        },
        "margins": {
            "d1_mean": d1.mean().item(), "d2_mean": d2.mean().item(),
            "rel_margin_p5": torch.quantile(rel_margin, .05).item(),
            "rel_margin_p50": torch.quantile(rel_margin, .50).item(),
            "rel_margin_p95": torch.quantile(rel_margin, .95).item(),
            "frac_below_0.05": (rel_margin < .05).float().mean().item(),
            "frac_below_0.10": (rel_margin < .10).float().mean().item(),
            "frac_below_0.25": (rel_margin < .25).float().mean().item(),
            "rel_quant_err_mean": quant_err.mean().item(),
            "rel_quant_err_p95": torch.quantile(quant_err, .95).item(),
        },
        "path_A": {
            "sigmas_x_latent_std": sigmas,
            "survival": survival,
            "sigma_at_99pct": s99,
            "sigma_at_95pct": s95,
        },
        "clip_frames": {
            "mean": float(frames.mean()), "p50": float(np.percentile(frames, 50)),
            "p95": float(np.percentile(frames, 95)), "max": int(frames.max()),
            "suggested_max_length": int(np.percentile(frames, 99) // 16 * 16 + 16),
        },
        "wav_manifest": manifest,
        "args": vars(args),
    }
    with open(os.path.join(args.out_dir, "test0.json"), "w") as f:
        json.dump(res, f, indent=2)
    np.savez_compressed(
        os.path.join(args.out_dir, "test0.npz"),
        rel_margin=rel_margin.cpu().numpy(), d1=d1.cpu().numpy(), d2=d2.cpu().numpy(),
        quant_err=quant_err.cpu().numpy(), frame_norm=frame_norm.cpu().numpy(),
        per_dim_std=per_dim_std.cpu().numpy(), code0=code0.cpu().numpy(),
        sigmas=np.array(sigmas), survival=np.array(survival), clip_frames=frames,
    )

    log("")
    log("=== SUMMARY — what to put in the config ===")
    log(f"  latent_std:  {latent_std:.5f}   (NOT ELF's 0.2)")
    log(f"  max_length:  {res['clip_frames']['suggested_max_length']}   "
        f"(p99 of frames/clip at {frame_rate:.0f} Hz)")
    log(f"  sigma_A@95:  {s95:.4f} x latent_std")
    log("")
    log("  GATE: if sigma_A@95 is a workable error budget and the margin-ambiguous")
    log("        fraction is small, the design clears. Path B (ELF's CE head) must")
    log("        then beat sigma_A@95 — measure after Test 2.")
    log(f"  wrote {args.out_dir}/test0.json and test0.npz")


if __name__ == "__main__":
    main()
