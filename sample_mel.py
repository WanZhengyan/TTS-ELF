#!/usr/bin/env python
"""Mel path: sample, vocode with Vocos, measure WER and UTMOS, build a listen page.

The mel path has no discrete codes, so the final z IS the generated mel. There is
no decode head and no codebook: the rollout ends, the mel is un-normalised, and
the pretrained Vocos vocoder turns it into a waveform.

Four tracks per clip, same structure as the codec path so the two are readable
side by side:

  orig      the original waveform, resampled to 24 kHz
  ceiling   original -> mel -> Vocos -> waveform. The upper bound for this path.
            Expected much higher than the codec path's 5.44 % WER, because mel at
            93.75 Hz x 100 bins keeps far more information than 40 codes/s.
  gen       the full rollout from noise, text-conditioned
  gen_uncond  text zeroed, as a control on how much the text is doing
"""
import argparse, base64, io, json, os, pickle, sys, html

import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train import (ELFTTS, LJCache, make_collate, Cfg,  # noqa: E402
                   get_sampling_steps, restore, net_to_v)
from torch.utils.data import DataLoader  # noqa: E402


def log(*a):
    print(*a, flush=True)


def wer_pair(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1); d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r), len(h)], len(r)


@torch.no_grad()
def rollout_mel(model, b, cfg, device, *, n_steps, cfg_scale, sampler, sde_gamma,
                schedule="logit_normal", use_text=True, seed=0):
    """Returns the generated mel in CACHE units (un-normalised), per clip."""
    torch.manual_seed(seed)
    x0 = (b["latent"].to(device) - cfg.latent_mean) / cfg.latent_std
    text = b["text"].to(device); prompt = b["prompt"].to(device)
    B = x0.shape[0]
    z = restore(torch.randn_like(x0) * cfg.denoiser_noise_scale, x0, prompt)
    x_pred = restore(torch.zeros_like(z), x0, prompt)
    ts = get_sampling_steps(n_steps, schedule, cfg.denoiser_p_mean,
                            cfg.denoiser_p_std, device)
    scb = torch.ones(B, device=device)
    drop = torch.zeros(B, device=device) if use_text else torch.ones(B, device=device)
    od = torch.ones(B, device=device)
    for i in range(len(ts) - 1):
        if sampler == "sde" and sde_gamma > 0:
            h = float(ts[i + 1] - ts[i])
            al = max(0.0, min(1.0, 1.0 - sde_gamma * h))
            z = restore(al * z + (1 - al) * torch.randn_like(z) * cfg.denoiser_noise_scale,
                        x0, prompt)
            t_cur = al * float(ts[i])
        else:
            t_cur = float(ts[i])
        t = torch.full((B,), t_cur, device=device)
        v_c, x_c = net_to_v(model(z, x_pred, text, t, self_cond_cfg_scale=scb,
                                  drop_text=drop), z, t)
        v_c = restore(v_c, torch.zeros_like(z), prompt); x_c = restore(x_c, x0, prompt)
        if cfg_scale != 1.0 and use_text:
            v_u, _ = net_to_v(model(z, x_pred, text, t, self_cond_cfg_scale=scb,
                                    drop_text=od), z, t)
            v_u = restore(v_u, torch.zeros_like(z), prompt)
            v = v_u + cfg_scale * (v_c - v_u)
        else:
            v = v_c
        z = restore(z + (float(ts[i + 1]) - t_cur) * v, x0, prompt)
        x_pred = x_c
    return z * cfg.latent_std + cfg.latent_mean      # back to log-mel



@torch.no_grad()
def mel_likeness(model, b, cfg, device, **kw):
    """Distinguish "did not learn" from "generated a different valid sample".

    ||z - x0|| cannot tell those apart: a perfect TTS model, asked for the same
    sentence, may legitimately produce different prosody and score badly against
    one specific recording. These three do separate them, and none needs a
    vocoder:

      diversity   two rollouts of the SAME text with different noise. Near zero
                  means mode collapse -- the model ignores z and emits one thing.
                  Real speech variation should be well above zero.
      lag1        temporal autocorrelation of the generated mel. Real mel at
                  93.75 Hz measures 0.953. A generated sequence with the right
                  lag1 is structurally mel-like even if it matches no recording.
      bin stats   per-mel-bin mean and std against the real distribution. Gets at
                  whether the output occupies the right region of mel space.
    """
    a = rollout_mel(model, b, cfg, device, use_text=True, seed=0, **kw)
    c = rollout_mel(model, b, cfg, device, use_text=True, seed=1, **kw)
    real = b["latent"].to(device)
    m = (b["valid"].to(device) * (1 - b["prompt"].to(device))).bool()

    def lag1(x):
        v = []
        for i in range(x.shape[0]):
            q = x[i][m[i]]
            if q.shape[0] < 3:
                continue
            u = torch.nn.functional.normalize(q - q.mean(0, keepdim=True), dim=-1)
            v.append((u[:-1] * u[1:]).sum(-1).mean().item())
        return float(np.mean(v)) if v else float("nan")

    def flat(x):
        return torch.cat([x[i][m[i]] for i in range(x.shape[0])], 0)

    fa, fc, fr = flat(a), flat(c), flat(real)
    div = (fa - fc).norm(dim=-1).mean().item() / (fr.norm(dim=-1).mean().item() + 1e-9)
    # spread of real mel between two DIFFERENT clips, as the scale diversity
    # should be compared against
    perm = torch.randperm(fr.shape[0], device=device)
    ref_div = (fr - fr[perm]).norm(dim=-1).mean().item() / (fr.norm(dim=-1).mean().item() + 1e-9)
    return {
        "diversity": div, "diversity_reference": ref_div,
        "lag1_gen": lag1(a), "lag1_real": lag1(real),
        "bin_mean_gen": float(fa.mean()), "bin_mean_real": float(fr.mean()),
        "bin_std_gen": float(fa.std()), "bin_std_real": float(fr.std()),
    }


def wav_b64(w, sr=24000):
    buf = io.BytesIO()
    torchaudio.save(buf, w.cpu(), sr, format="wav", encoding="PCM_S", bits_per_sample=16)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "h_mel_xs", "ckpt.pt"))
    ap.add_argument("--cache", default=os.path.join(HERE, "cache_mel"))
    ap.add_argument("--lj_root", default=os.path.join(HERE, "data", "LJSpeech-1.1"))
    ap.add_argument("--weights", default="ema")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--n_wav", type=int, default=6)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--n_steps", type=int, default=32)
    ap.add_argument("--cfg_scale", type=float, default=2.0)
    ap.add_argument("--sampler", default="sde")
    ap.add_argument("--sde_gamma", type=float, default=1.0)
    ap.add_argument("--asr", default="openai/whisper-large-v3")
    ap.add_argument("--page", default=os.path.join(HERE, "listen_mel.html"))
    ap.add_argument("--out", default=os.path.join(HERE, "out", "metrics_mel.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ta = ck["args"]
    cfg = Cfg(**{k: v for k, v in ck["cfg"].items() if k in Cfg.__dataclass_fields__})
    log(f"ckpt {ta['model']} @ {ta['steps']} steps ({args.weights}) | "
        f"latent_mean {cfg.latent_mean:+.4f} std {cfg.latent_std:.4f}")

    from vocos import Vocos
    voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
    log("Vocos charactr/vocos-mel-24khz loaded")

    from transformers import pipeline
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    asr = pipeline("automatic-speech-recognition", model=args.asr,
                   device=0 if device.type == "cuda" else -1,
                   torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    norm = EnglishTextNormalizer({})
    try:
        um = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                            trust_repo=True).to(device).eval()
    except Exception as e:
        log(f"UTMOS unavailable: {e}"); um = None

    ds = LJCache(args.cache, "test", seed=ta["seed"])
    meta = ds.meta
    id2text = {}
    with open(os.path.join(args.lj_root, "metadata.csv"), encoding="utf-8") as f:
        for line in f:
            p = line.strip().split("|")
            if len(p) >= 3:
                id2text[p[0]] = p[2]
    coll = make_collate(ta["max_length"], (0.0, 0.0), ta["seed"], fixed_prompt=0.0,
                        dim=meta["dim"])
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0,
                    collate_fn=coll)
    model = ELFTTS(ta["model"], meta["char_vocab_size"], meta["audio_vocab_size"],
                   dim=meta["dim"], max_length=ta["max_length"],
                   text_norm=not ta.get("no_text_norm", False)).to(device)
    model.load_state_dict(ck[args.weights]); model.eval()
    rs = torchaudio.transforms.Resample(22050, 24000).to(device)

    tracks = ["orig", "ceiling", "gen", "gen_uncond"]
    errs = {t: 0 for t in tracks}; lens = {t: 0 for t in tracks}
    mos = {t: [] for t in tracks}
    rows, done = [], 0

    for b in dl:
        if done >= args.n:
            break
        mel_gen = rollout_mel(model, b, cfg, device, n_steps=args.n_steps,
                              cfg_scale=args.cfg_scale, sampler=args.sampler,
                              sde_gamma=args.sde_gamma, use_text=True)
        mel_unc = rollout_mel(model, b, cfg, device, n_steps=args.n_steps,
                              cfg_scale=1.0, sampler=args.sampler,
                              sde_gamma=args.sde_gamma, use_text=False)
        mel_true = b["latent"].to(device)

        for i in range(mel_gen.shape[0]):
            if done >= args.n:
                break
            uid = b["id"][i]; T = int(b["T"][i])
            ref = norm(id2text.get(uid, ""))
            if not ref:
                continue
            w, _ = torchaudio.load(os.path.join(args.lj_root, "wavs", uid + ".wav"))
            g = rs(w.to(device)); g = g.mean(0, keepdim=True) if g.shape[0] > 1 else g
            waves = {
                "orig": g,
                "ceiling": voc.decode(mel_true[i:i + 1, :T].transpose(1, 2)),
                "gen": voc.decode(mel_gen[i:i + 1, :T].transpose(1, 2)),
                "gen_uncond": voc.decode(mel_unc[i:i + 1, :T].transpose(1, 2)),
            }
            for t in tracks:
                a = waves[t].squeeze(0).float().clamp(-1, 1)
                out = asr({"sampling_rate": 24000, "raw": a.cpu().numpy()},
                          generate_kwargs={"language": "english"})
                e, n = wer_pair(ref, norm(out["text"]))
                errs[t] += e; lens[t] += n
                if um is not None:
                    mos[t].append(float(um(a.unsqueeze(0), 24000).item()))
            if len(rows) < args.n_wav:
                rows.append({"id": uid, "text": id2text.get(uid, ""), "T": T,
                             "dur": T / meta["frame_rate"],
                             "b64": {t: wav_b64(waves[t].squeeze(0).unsqueeze(0).cpu())
                                     for t in tracks}})
            done += 1
            if done % 10 == 0:
                log(f"  {done}/{args.n}  " + "  ".join(
                    f"{t} {100*errs[t]/max(lens[t],1):.1f}" for t in tracks))

    ml = mel_likeness(model, b, cfg, device, n_steps=args.n_steps,
                      cfg_scale=args.cfg_scale, sampler=args.sampler,
                      sde_gamma=args.sde_gamma)
    log("")
    log("=== is it generating at all? (no vocoder needed) ===")
    log(f"  diversity: two rollouts of the same text differ by {ml['diversity']:.3f}")
    log(f"             two DIFFERENT real clips differ by      {ml['diversity_reference']:.3f}")
    log(f"             -> {'collapsed, ignores the noise' if ml['diversity'] < 0.05*ml['diversity_reference'] else 'the noise does change the output'}")
    log(f"  lag1 autocorrelation  generated {ml['lag1_gen']:.3f}  real {ml['lag1_real']:.3f}")
    log(f"             -> {'structurally mel-like' if abs(ml['lag1_gen']-ml['lag1_real'])<0.15 else 'NOT mel-like in time'}")
    log(f"  mel-bin mean  gen {ml['bin_mean_gen']:+.3f}  real {ml['bin_mean_real']:+.3f}")
    log(f"  mel-bin std   gen {ml['bin_std_gen']:.3f}  real {ml['bin_std_real']:.3f}")
    log("")
    log(f"=== mel path, {done} held-out clips, pure TTS (prompt=0) ===")
    log(f"{'track':<12} {'WER %':>8} {'UTMOS':>8}")
    res = {"n": done, "model": ta["model"], "steps": ta["steps"], "tracks": {}}
    for t in tracks:
        wv = 100 * errs[t] / max(lens[t], 1)
        uv = float(np.mean(mos[t])) if mos[t] else float("nan")
        log(f"{t:<12} {wv:>8.2f} {uv:>8.3f}")
        res["tracks"][t] = {"wer": wv, "utmos": uv}
    log("")
    log("compare with the codec path on the same split: gt 2.01 / ceiling 5.44 /")
    log("teacher-forced 5.27 / generated 99.17 WER.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    res["mel_likeness"] = ml
    json.dump(res, open(args.out, "w"), indent=2)

    LBL = [("orig", "Original", "the real recording"),
           ("ceiling", "Mel ceiling", "original → mel → Vocos. The upper bound for this path."),
           ("gen", "Generated", "full rollout from noise, text-conditioned"),
           ("gen_uncond", "Unconditional", "text zeroed — how much is the text doing?")]
    arts = []
    for r in rows:
        tr = "".join(
            f'<div class="tk"><div><b>{html.escape(l)}</b><span>{html.escape(d)}</span></div>'
            f'<audio controls preload="none" src="data:audio/wav;base64,{r["b64"][k]}">'
            f'</audio></div>' for k, l, d in LBL)
        arts.append(f'<article><header><h3>{html.escape(r["id"])}</h3>'
                    f'<span>{r["dur"]:.1f} s · {r["T"]} frames</span></header>'
                    f'<p>{html.escape(r["text"])}</p>{tr}</article>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Mel Path Samples</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Serif:ital@1&family=IBM+Plex+Mono&display=swap">
<style>
:root{{color-scheme:light;--g:#f6f7f9;--s:#fff;--s2:#eceff4;--i:#171b22;--i2:#4a5260;--i3:#79818f;--r:#d7dbe3;--c:#0d7d72}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{color-scheme:dark;--g:#0e1116;--s:#161b22;--s2:#1d232b;--i:#e7eaef;--i2:#a9b1be;--i3:#7d8591;--r:#2a313b;--c:#45bdad}}}}
:root[data-theme=dark]{{color-scheme:dark;--g:#0e1116;--s:#161b22;--s2:#1d232b;--i:#e7eaef;--i2:#a9b1be;--i3:#7d8591;--r:#2a313b;--c:#45bdad}}
html,body{{margin:0}}body{{background:var(--g);color:var(--i);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:15px;line-height:1.6}}
.w{{max-width:820px;margin:0 auto;padding-left:18px;padding-right:18px;padding-block:0 60px}}
h1{{font-size:clamp(25px,5vw,34px);margin:36px 0 10px;font-weight:600;letter-spacing:-.02em}}
.lede{{color:var(--i2);max-width:60ch;margin:0 0 20px}}
article{{border:1px solid var(--r);background:var(--s);margin:20px 0 0}}
article>header{{padding:11px 14px;border-bottom:1px solid var(--r);background:var(--s2);display:flex;gap:14px;align-items:baseline;flex-wrap:wrap}}
h3{{margin:0;font-family:"IBM Plex Mono",monospace;font-size:13.5px}}
header span{{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--i3)}}
article p{{margin:0;padding:11px 14px;font-family:"IBM Plex Serif",Georgia,serif;font-style:italic;color:var(--i2);border-bottom:1px solid var(--r)}}
.tk{{padding:9px 14px;border-bottom:1px solid var(--r);display:grid;grid-template-columns:minmax(0,240px) 1fr;gap:6px 16px;align-items:center}}
.tk:last-child{{border-bottom:none}}
.tk div{{display:flex;flex-direction:column}}
.tk b{{font-size:13.5px}}.tk span{{font-size:11.5px;color:var(--i3);line-height:1.4}}
audio{{width:100%;height:34px}}
@media(max-width:620px){{.tk{{grid-template-columns:1fr}}}}
</style></head><body><div class="w">
<h1>Mel Path Samples</h1>
<p class="lede">{html.escape(ta['model'])} @ {ta['steps']:,} steps, log-mel at 93.75 Hz,
vocoded by pretrained Vocos. Compare <b>Generated</b> against <b>Mel ceiling</b>: the gap is the
model's, everything above it is the vocoder's.</p>
{''.join(arts)}
</div></body></html>"""
    open(args.page, "w").write(page)
    log(f"wrote {args.out} and {args.page}")


if __name__ == "__main__":
    main()
