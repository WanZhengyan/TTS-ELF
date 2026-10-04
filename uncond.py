#!/usr/bin/env python
"""Unconditional generation: zero BOTH the text and self-conditioning channels.

The question this settles. Generation currently fails (WER 99.2%) while the
discretization head succeeds (teacher-forced WER 5.3%, matching the codec
ceiling). Two very different causes are still consistent with that:

  (a) the flow works, but the text conditioning never gets through. The earlier
      finding that the text block enters ~27x weaker in magnitude than z points
      here. If so, unconditional generation should produce speech-LIKE audio --
      babble, wrong words, but recognisably voice.

  (b) the flow itself never learned the data manifold. Then unconditional
      generation is noise too, and text is beside the point.

Zeroing both channels is in-distribution: training zeroes text on 20% of
examples (label_drop_prob) and zeroes the self-cond block on 50% of denoiser
rows, so both-zero occurs on ~10% of denoiser rows.

Reported per condition:
  z_err      ||z_final - x0||, though for unconditional output matching a
             specific clip is NOT the goal -- read it only as "did the
             trajectory land anywhere near the data manifold".
  utmos      the real signal here. Does it sound like speech at all?
             Codec ceiling is 3.91, pure noise is ~1.3.
  code_ent   entropy of the predicted code histogram, in bits (max 12).
             Real speech uses the codebook unevenly; a collapsed model emits a
             few codes over and over, a noise model emits uniform codes.
  uniq       distinct codes used, out of 4096.
"""
import argparse, base64, io, json, os, sys, html

import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "third_party", "WavTokenizer"))
sys.path.insert(0, HERE)
from train import (ELFTTS, LJCache, make_collate, Cfg,  # noqa: E402
                   get_sampling_steps, restore, net_to_v)
from decoder.pretrained import WavTokenizer  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def log(*a):
    print(*a, flush=True)


@torch.no_grad()
def rollout(model, b, cfg, device, *, use_text, use_sc, cfg_scale,
            n_steps=32, schedule="logit_normal", seed=0):
    torch.manual_seed(seed)
    x0 = b["latent"].to(device) / cfg.latent_std
    text = b["text"].to(device)
    prompt = b["prompt"].to(device)
    m = b["valid"].to(device) * (1 - prompt)
    B = x0.shape[0]
    nrm = lambda a: ((a.norm(dim=-1) * m).sum() / m.sum().clamp(min=1)).item()

    z = restore(torch.randn_like(x0) * cfg.denoiser_noise_scale, x0, prompt)
    zero_sc = restore(torch.zeros_like(z), x0, prompt)
    x_pred = zero_sc
    ts = get_sampling_steps(n_steps, schedule, cfg.denoiser_p_mean,
                            cfg.denoiser_p_std, device)
    scb = torch.ones(B, device=device)
    drop = torch.zeros(B, device=device) if use_text else torch.ones(B, device=device)
    od = torch.ones(B, device=device)

    for i in range(len(ts) - 1):
        t = ts[i].expand(B)
        sc_in = x_pred if use_sc else zero_sc
        v_c, x_c = net_to_v(model(z, sc_in, text, t, self_cond_cfg_scale=scb,
                                  drop_text=drop), z, t)
        v_c = restore(v_c, torch.zeros_like(z), prompt)
        x_c = restore(x_c, x0, prompt)
        if cfg_scale != 1.0 and use_text:
            v_u, _ = net_to_v(model(z, sc_in, text, t, self_cond_cfg_scale=scb,
                                    drop_text=od), z, t)
            v_u = restore(v_u, torch.zeros_like(z), prompt)
            v = v_u + cfg_scale * (v_c - v_u)
        else:
            v = v_c
        z = restore(z + (ts[i + 1] - ts[i]) * v, x0, prompt)
        x_pred = x_c

    t1 = torch.ones(B, device=device)
    _, logits = model(z, torch.zeros_like(z), text, t1,
                      self_cond_cfg_scale=scb, decoder_step_active=True,
                      drop_text=drop)
    return logits.argmax(-1), nrm(z - x0), m


def code_stats(codes, mask, K=4096):
    """Entropy in bits and distinct-code count over the masked positions."""
    c = codes[mask > 0].flatten()
    if c.numel() == 0:
        return float("nan"), 0
    h = torch.bincount(c, minlength=K).float()
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * p.log2()).sum()), int((h > 0).sum())


def wav_b64(wav, sr=24000):
    buf = io.BytesIO()
    torchaudio.save(buf, wav.cpu(), sr, format="wav",
                    encoding="PCM_S", bits_per_sample=16)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "d_xs_dp02", "ckpt.pt"))
    ap.add_argument("--cache", default=os.path.join(HERE, "cache"))
    ap.add_argument("--codec_cfg", default=os.path.join(
        HERE, "third_party", "WavTokenizer", "configs",
        "wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"))
    ap.add_argument("--codec_ckpt", default=os.path.join(
        HERE, "ckpt", "wavtokenizer_large_unify_600_24k.ckpt"))
    ap.add_argument("--weights", default="ema")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n_wav", type=int, default=4)
    ap.add_argument("--page", default=os.path.join(HERE, "listen_uncond.html"))
    ap.add_argument("--out", default=os.path.join(HERE, "out", "uncond.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ta = ck["args"]
    cfg = Cfg(**{k: v for k, v in ck["cfg"].items() if k in Cfg.__dataclass_fields__})
    log(f"ckpt {ta['model']} @ {ta['steps']} steps, {args.weights}")

    val = LJCache(args.cache, "val", seed=ta["seed"])
    meta = val.meta
    # prompt = 0 everywhere: no ground-truth audio leaks in at all
    coll = make_collate(ta["max_length"], (0.0, 0.0), ta["seed"], fixed_prompt=0.0)
    b = next(iter(DataLoader(val, batch_size=args.batch, shuffle=False,
                             num_workers=0, collate_fn=coll)))

    model = ELFTTS(ta["model"], meta["char_vocab_size"], meta["audio_vocab_size"],
                   dim=512, max_length=ta["max_length"],
                   cb_skip=ta.get("cb_skip", False)).to(device)
    model.load_state_dict(ck[args.weights]); model.eval()

    wt = WavTokenizer.from_pretrained0802(args.codec_cfg, args.codec_ckpt).to(device).eval()
    vq = wt.feature_extractor.encodec.quantizer.vq.layers[0]
    bw = torch.tensor([0], device=device)
    try:
        utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                               trust_repo=True).to(device).eval()
    except Exception as e:
        log(f"  UTMOS unavailable: {e}")
        utmos = None

    conds = [
        ("text + SC  (the system)",        True,  True,  2.0),
        ("text, no SC",                    True,  False, 2.0),
        ("text, no CFG",                   True,  True,  1.0),
        ("UNCONDITIONAL (text=0, SC=0)",   False, False, 1.0),
        ("UNCONDITIONAL (text=0, SC on)",  False, True,  1.0),
    ]
    code_true = b["code"].to(device)
    results, audios = [], {}
    log("")
    log(f"{'condition':<32} {'||z-x0||':>9} {'UTMOS':>7} {'code bits':>10} {'uniq':>6}")
    log("-" * 72)
    for name, ut, us, cg in conds:
        pred, zerr, m = rollout(model, b, cfg, device, use_text=ut, use_sc=us, cfg_scale=cg)
        ent, uniq = code_stats(pred, m)
        wavs = [wt.decode(vq.decode(pred[i:i+1, :int(b["T"][i])]), bandwidth_id=bw)
                for i in range(min(args.n_wav, pred.shape[0]))]
        mos = float(np.mean([utmos(w.squeeze(0).float().clamp(-1, 1).unsqueeze(0),
                                   24000).item() for w in wavs])) if utmos else float("nan")
        log(f"{name:<32} {zerr:>9.3f} {mos:>7.3f} {ent:>10.3f} {uniq:>6}")
        results.append({"condition": name, "use_text": ut, "use_sc": us, "cfg": cg,
                        "z_err": zerr, "utmos": mos, "code_bits": ent, "uniq_codes": uniq})
        audios[name] = wavs

    # references
    ref_w = [wt.decode(vq.decode(code_true[i:i+1, :int(b["T"][i])]), bandwidth_id=bw)
             for i in range(min(args.n_wav, code_true.shape[0]))]
    ent_r, uniq_r = code_stats(code_true, b["valid"].to(device))
    mos_r = float(np.mean([utmos(w.squeeze(0).float().clamp(-1, 1).unsqueeze(0),
                                 24000).item() for w in ref_w])) if utmos else float("nan")
    log(f"{'codec ceiling (reference)':<32} {0.0:>9.3f} {mos_r:>7.3f} {ent_r:>10.3f} {uniq_r:>6}")
    audios["codec ceiling (reference)"] = ref_w
    results.append({"condition": "codec ceiling (reference)", "z_err": 0.0,
                    "utmos": mos_r, "code_bits": ent_r, "uniq_codes": uniq_r})

    log("")
    log("HOW TO READ IT")
    log("  If the UNCONDITIONAL rows sound like speech (babble, wrong words, but voice)")
    log("  and their UTMOS is near the ceiling, the flow works and the text pathway is")
    log("  the broken part -- consistent with the text block entering ~27x weaker.")
    log("  If they are noise too, the flow never learned the manifold and text is")
    log("  beside the point.")
    log("  Code entropy: the ceiling's value is what real speech uses. Far below it")
    log("  means collapse onto a few codes; near 12 bits means uniform = noise.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"ckpt": args.ckpt, "model": ta["model"], "steps": ta["steps"],
               "results": results}, open(args.out, "w"), indent=2)

    # ---- page
    secs = []
    for name, wavs in audios.items():
        r = next(x for x in results if x["condition"] == name)
        tr = "".join(
            f'<audio controls preload="none" src="data:audio/wav;base64,{wav_b64(w)}"></audio>'
            for w in wavs)
        secs.append(f"""<article><header><h3>{html.escape(name)}</h3>
<div class="meta"><span>UTMOS {r['utmos']:.3f}</span><span>code {r['code_bits']:.2f} bits</span>
<span>{r['uniq_codes']} distinct codes</span></div></header>
<div class="tracks">{tr}</div></article>""")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Unconditional Probe</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono&display=swap">
<style>
:root{{color-scheme:light;--g:#f6f7f9;--s:#fff;--s2:#eceff4;--i:#171b22;--i2:#4a5260;--i3:#79818f;--r:#d7dbe3;--c:#0d7d72}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{color-scheme:dark;--g:#0e1116;--s:#161b22;--s2:#1d232b;--i:#e7eaef;--i2:#a9b1be;--i3:#7d8591;--r:#2a313b;--c:#45bdad}}}}
:root[data-theme=dark]{{color-scheme:dark;--g:#0e1116;--s:#161b22;--s2:#1d232b;--i:#e7eaef;--i2:#a9b1be;--i3:#7d8591;--r:#2a313b;--c:#45bdad}}
html,body{{margin:0}}body{{background:var(--g);color:var(--i);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:15px;line-height:1.6}}
.w{{max-width:820px;margin:0 auto;padding-left:18px;padding-right:18px;padding-block:0 60px}}
h1{{font-size:clamp(25px,5vw,34px);margin:36px 0 12px;font-weight:600;letter-spacing:-.02em}}
.lede{{color:var(--i2);max-width:60ch;margin:0 0 8px}}
.note{{border-left:2px solid var(--c);padding:10px 14px;background:var(--s2);font-size:13.5px;margin:20px 0;border-radius:0 3px 3px 0}}
article{{border:1px solid var(--r);background:var(--s);margin:20px 0 0}}
article>header{{padding:11px 14px;border-bottom:1px solid var(--r);background:var(--s2)}}
h3{{margin:0 0 4px;font-family:"IBM Plex Mono",monospace;font-size:13.5px;font-weight:600}}
.meta{{display:flex;gap:14px;flex-wrap:wrap;font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--i3)}}
.tracks{{padding:10px 14px;display:flex;flex-direction:column;gap:8px}}
audio{{width:100%;height:34px}}
</style></head><body><div class="w">
<h1>Unconditional Probe</h1>
<p class="lede">{html.escape(ta['model'])} @ {ta['steps']:,} steps. Each row is the same sampler with
different inputs switched off. No ground-truth audio anywhere (prompt = 0).</p>
<div class="note"><b>What to listen for.</b> If the UNCONDITIONAL rows sound like speech — babble,
wrong words, but recognisably a voice — then the flow works and the text pathway is the broken part.
If they are noise, the flow never learned the data manifold. Compare everything against
<b>codec ceiling</b>, the last row, which is what the tokenizer alone can do.</div>
{''.join(secs)}
</div></body></html>"""
    open(args.page, "w").write(page)
    log("")
    log(f"wrote {args.out} and {args.page}")


if __name__ == "__main__":
    main()
