#!/usr/bin/env python
"""Generate audio from a trained ELF-TTS checkpoint and build a playable page.

Produces, for each held-out clip:
  _orig      the original LJSpeech waveform, resampled to 24 kHz
  _ceiling   original -> codec encode -> VQ -> codec decode. The upper bound:
             no generative model can sound better than this.
  _gen       full ODE rollout + CE decode -> codes -> codec decoder
  _gen_tf    teacher-forced control: the TRUE x0 fed to the CE head, then decoded.
             Isolates the discretizer from the flow — if this sounds fine and _gen
             does not, the flow is the problem, not the head.

Then writes listen.html with every clip embedded as a data URI, so it plays in any
browser with no server and no file copying beyond that one file.
"""
import argparse, base64, io, json, os, sys, html

import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
# WavTokenizer ships its own train.py, so our package dir must win: insert it last
# so it ends up at sys.path[0].
sys.path.insert(0, os.path.join(HERE, "third_party", "WavTokenizer"))
sys.path.insert(0, HERE)
from train import (ELFTTS, LJCache, make_collate, Cfg, sample as elf_sample,  # noqa: E402
                   restore)
from decoder.pretrained import WavTokenizer  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def log(*a):
    print(*a, flush=True)


def wav_to_b64(wav, sr=24000):
    """16-bit PCM WAV as a base64 data URI."""
    buf = io.BytesIO()
    torchaudio.save(buf, wav.cpu(), sr, format="wav",
                    encoding="PCM_S", bits_per_sample=16)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "test2", "ckpt.pt"))
    ap.add_argument("--cache", default=os.path.join(HERE, "cache"))
    ap.add_argument("--lj_root", default=os.path.join(HERE, "data", "LJSpeech-1.1"))
    ap.add_argument("--codec_cfg", default=os.path.join(
        HERE, "third_party", "WavTokenizer", "configs",
        "wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"))
    ap.add_argument("--codec_ckpt", default=os.path.join(
        HERE, "ckpt", "wavtokenizer_large_unify_600_24k.ckpt"))
    ap.add_argument("--weights", default="ema")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--n_steps", type=int, default=32)
    ap.add_argument("--cfg_scale", type=float, default=2.0)
    ap.add_argument("--schedule", default="logit_normal")
    ap.add_argument("--prompt_frac", type=float, default=0.3)
    ap.add_argument("--out_dir", default=os.path.join(HERE, "out", "listen"))
    ap.add_argument("--page", default=os.path.join(HERE, "listen.html"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ta = ck["args"]
    cfg = Cfg(**{k: v for k, v in ck["cfg"].items() if k in Cfg.__dataclass_fields__})
    log(f"ckpt {args.ckpt} | {ta['model']} @ {ta['steps']} steps | weights {args.weights}")

    # codec
    wt = WavTokenizer.from_pretrained0802(args.codec_cfg, args.codec_ckpt).to(device).eval()
    vq = wt.feature_extractor.encodec.quantizer.vq.layers[0]
    bw = torch.tensor([0], device=device)

    # data — the held-out test split, with a fixed prompt fraction
    ds = LJCache(args.cache, "test", seed=ta["seed"])
    meta = ds.meta
    id2text = {r["id"]: None for r in meta["records"]}
    with open(os.path.join(args.lj_root, "metadata.csv"), encoding="utf-8") as f:
        for line in f:
            p = line.strip().split("|")
            if len(p) >= 3 and p[0] in id2text:
                id2text[p[0]] = p[2]

    coll = make_collate(ta["max_length"], (args.prompt_frac, args.prompt_frac),
                        ta["seed"], fixed_prompt=args.prompt_frac)
    dl = DataLoader(ds, batch_size=args.n, shuffle=False, num_workers=0, collate_fn=coll)
    b = next(iter(dl))

    model = ELFTTS(ta["model"], meta["char_vocab_size"], meta["audio_vocab_size"],
                   dim=512, max_length=ta["max_length"],
                   cb_skip=ta.get("cb_skip", False)).to(device)
    model.load_state_dict(ck[args.weights])
    model.eval()

    # ---- generation
    log(f"sampling: {args.n_steps} steps, {args.schedule}, cfg {args.cfg_scale}, "
        f"prompt {args.prompt_frac:.0%}")
    pred, z_final, nfe = elf_sample(model, b, cfg, device, n_steps=args.n_steps,
                                    cfg_scale=args.cfg_scale, schedule=args.schedule)
    log(f"  NFE {nfe}")

    # ---- teacher-forced control: feed the TRUE x0 to the CE head
    x0 = b["latent"].to(device) / cfg.latent_std
    text = b["text"].to(device)
    B = x0.shape[0]
    t1 = torch.ones(B, device=device)
    sc = torch.ones(B, device=device)
    _, lg_tf = model(x0, torch.zeros_like(x0), text, t1,
                     self_cond_cfg_scale=sc, decoder_step_active=True)
    pred_tf = lg_tf.argmax(-1)

    code_true = b["code"].to(device)
    msk = b["valid"].to(device) * (1 - b["prompt"].to(device))
    accg = (((pred == code_true).float() * msk).sum() / msk.sum()).item()
    acct = (((pred_tf == code_true).float() * msk).sum() / msk.sum()).item()
    log(f"  code acc — generated {accg*100:.2f}%  teacher-forced {acct*100:.2f}%")

    resample = torchaudio.transforms.Resample(22050, 24000).to(device)

    def decode_codes(codes_1T):
        q = vq.decode(codes_1T)                        # (1, D, T)
        return wt.decode(q, bandwidth_id=bw)           # (1, S)

    rows = []
    for i in range(min(args.n, len(b["id"]))):
        uid = b["id"][i]
        T = int(b["T"][i]); P = int(b["P"][i])
        text_s = id2text.get(uid) or ""

        # original
        w, sr = torchaudio.load(os.path.join(args.lj_root, "wavs", uid + ".wav"))
        orig = resample(w.to(device))
        if orig.shape[0] > 1:
            orig = orig.mean(0, keepdim=True)

        ceiling = decode_codes(code_true[i:i + 1, :T])
        gen = decode_codes(pred[i:i + 1, :T])
        gen_tf = decode_codes(pred_tf[i:i + 1, :T])
        # Frames [0,P) of `gen` are NOT generated: sampling pins them to the true
        # x0 via restore_cond at every step, so the CE head decodes ground truth
        # there. Only [P,T) is the model's own output.
        gen_only = (decode_codes(pred[i:i + 1, P:T]) if P < T and P > 0 else gen)

        for tag, wav in (("orig", orig), ("ceiling", ceiling),
                         ("gen", gen), ("gen_only", gen_only), ("gen_tf", gen_tf)):
            torchaudio.save(os.path.join(args.out_dir, f"{uid}_{tag}.wav"),
                            wav.cpu(), 24000, encoding="PCM_S", bits_per_sample=16)

        a_i = (((pred[i, :T] == code_true[i, :T]).float() * msk[i, :T]).sum()
               / msk[i, :T].sum().clamp(min=1)).item()
        rows.append({
            "id": uid, "text": text_s, "T": T, "P": P,
            "dur": T / meta["frame_rate"], "code_acc": a_i,
            "b64": {t: wav_to_b64(w) for t, w in
                    (("orig", orig), ("ceiling", ceiling), ("gen", gen),
                     ("gen_only", gen_only), ("gen_tf", gen_tf))},
        })
        log(f"  {uid}  T={T} P={P}  {T/meta['frame_rate']:.1f}s  code acc {a_i*100:.2f}%")

    # ---------------------------------------------------------------- page
    info = {
        "ckpt": os.path.relpath(args.ckpt, HERE), "model": ta["model"],
        "steps": ta["steps"], "weights": args.weights,
        "n_steps": args.n_steps, "schedule": args.schedule,
        "cfg_scale": args.cfg_scale, "prompt_frac": args.prompt_frac, "nfe": nfe,
        "code_acc_gen": accg, "code_acc_tf": acct,
    }
    build_page(rows, info, args.page)
    log("")
    log(f"wrote {len(rows)*4} wavs to {args.out_dir}")
    log(f"wrote {args.page}  <- open this in a browser, audio is embedded")
    with open(os.path.join(HERE, "out", "listen_meta.json"), "w") as f:
        json.dump(info, f, indent=2)


TRACKS = [
    ("orig", "Original", "LJSpeech ground truth, resampled to 24 kHz", "ok"),
    ("ceiling", "Codec ceiling", "original → encode → VQ → decode. No model can beat this.", "ok"),
    ("gen", "Prompt + generated",
     "the opening prompt frames are pinned to the TRUE latents by restore_cond, so that "
     "part is not generated — it decodes ground truth. Only after the prompt is it the model.", "gen"),
    ("gen_only", "Generated only",
     "frames after the prompt, nothing else. This is the model's actual output.", "gen"),
    ("gen_tf", "Teacher-forced", "true x₀ fed to the CE head — isolates the discretizer", "tf"),
]


def build_page(rows, info, path):
    esc = html.escape
    parts = []
    for r in rows:
        tracks = []
        for key, label, desc, kind in TRACKS:
            tracks.append(f"""
        <div class="track {kind}">
          <div class="tl"><span class="lbl">{esc(label)}</span><span class="dsc">{esc(desc)}</span></div>
          <audio controls preload="none" src="data:audio/wav;base64,{r['b64'][key]}"></audio>
        </div>""")
        parts.append(f"""
      <article>
        <header>
          <h3>{esc(r['id'])}</h3>
          <div class="meta"><span>{r['dur']:.1f} s</span><span>{r['T']} frames</span>
            <span>prompt {r['P']} fr</span><span>code acc {r['code_acc']*100:.2f}%</span></div>
        </header>
        <p class="txt">{esc(r['text'])}</p>
        {''.join(tracks)}
      </article>""")

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ELF-TTS Samples</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:ital@0;1&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
*{{box-sizing:border-box}}
:root{{color-scheme:light;--ground:#f6f7f9;--surface:#fff;--surface-2:#eceff4;--ink:#171b22;
--ink-2:#4a5260;--ink-3:#79818f;--rule:#d7dbe3;--rule-2:#b7bfcb;--clean:#0d7d72;--clean-bg:#e2f2f0;
--noise:#a85d08;--noise-bg:#f8eddd;--txt:#3b5490;--txt-bg:#e5eaf6;--warn:#a0262b;--warn-bg:#f8e7e7;
--sans:"IBM Plex Sans",system-ui,sans-serif;--serif:"IBM Plex Serif",Georgia,serif;
--mono:"IBM Plex Mono",Menlo,monospace}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{color-scheme:dark;
--ground:#0e1116;--surface:#161b22;--surface-2:#1d232b;--ink:#e7eaef;--ink-2:#a9b1be;--ink-3:#7d8591;
--rule:#2a313b;--rule-2:#3c4550;--clean:#45bdad;--clean-bg:#0f2c2a;--noise:#e39c4c;--noise-bg:#33240f;
--txt:#8099d8;--txt-bg:#1a2340;--warn:#e4807f;--warn-bg:#331a1b}}}}
:root[data-theme="dark"]{{color-scheme:dark;--ground:#0e1116;--surface:#161b22;--surface-2:#1d232b;
--ink:#e7eaef;--ink-2:#a9b1be;--ink-3:#7d8591;--rule:#2a313b;--rule-2:#3c4550;--clean:#45bdad;
--clean-bg:#0f2c2a;--noise:#e39c4c;--noise-bg:#33240f;--txt:#8099d8;--txt-bg:#1a2340;
--warn:#e4807f;--warn-bg:#331a1b}}
html,body{{margin:0;padding:0}}
body{{background:var(--ground);color:var(--ink);font-family:var(--sans);font-size:15px;line-height:1.6}}
.wrap{{max-width:860px;margin:0 auto;padding-left:18px;padding-right:18px;padding-block:0 64px}}
header.mast{{padding-block:36px 20px;border-bottom:2px solid var(--ink)}}
.eyebrow{{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;
color:var(--ink-3);margin:0 0 12px}}
h1{{font-size:clamp(26px,5vw,36px);line-height:1.1;letter-spacing:-.02em;font-weight:600;margin:0 0 12px}}
.lede{{font-family:var(--serif);font-size:16px;color:var(--ink-2);max-width:60ch;margin:0}}
.spec{{display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));gap:1px;background:var(--rule);
border:1px solid var(--rule);margin:24px 0 0}}
.spec div{{background:var(--surface);padding:10px 12px}}
.spec dt{{font-family:var(--mono);font-size:10px;letter-spacing:.07em;text-transform:uppercase;
color:var(--ink-3);margin:0 0 3px}}
.spec dd{{margin:0;font-family:var(--mono);font-size:13.5px;font-weight:500;font-variant-numeric:tabular-nums}}
.guide{{margin:26px 0 0;border:1px solid var(--rule);background:var(--surface)}}
.guide > div{{padding:11px 14px;border-top:1px solid var(--rule);font-size:13.5px;color:var(--ink-2)}}
.guide > div:first-child{{border-top:none}}
.guide b{{color:var(--ink)}}
.warn{{margin:22px 0 0;border-left:2px solid var(--warn);background:var(--warn-bg);padding:11px 14px;
font-size:13.5px;border-radius:0 3px 3px 0}}
article{{margin:26px 0 0;border:1px solid var(--rule);background:var(--surface)}}
article > header{{padding:12px 14px;border-bottom:1px solid var(--rule);background:var(--surface-2);
display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}}
article h3{{margin:0;font-family:var(--mono);font-size:14px;font-weight:600}}
.meta{{display:flex;gap:14px;flex-wrap:wrap;font-family:var(--mono);font-size:11px;color:var(--ink-3)}}
.txt{{margin:0;padding:12px 14px;font-family:var(--serif);font-style:italic;color:var(--ink-2);
border-bottom:1px solid var(--rule)}}
.track{{padding:10px 14px;border-bottom:1px solid var(--rule);display:grid;
grid-template-columns:minmax(0,250px) 1fr;gap:6px 16px;align-items:center}}
.track:last-child{{border-bottom:none}}
@media (max-width:620px){{.track{{grid-template-columns:1fr}}}}
.tl{{display:flex;flex-direction:column;gap:2px}}
.lbl{{font-size:13.5px;font-weight:600}}
.dsc{{font-size:11.5px;color:var(--ink-3);line-height:1.4}}
.track.ok .lbl{{color:var(--clean)}}
.track.gen .lbl{{color:var(--noise)}}
.track.tf .lbl{{color:var(--txt)}}
audio{{width:100%;height:34px}}
</style></head><body><div class="wrap">
<header class="mast">
<p class="eyebrow">voiceflow · {esc(info['model'])} @ {info['steps']:,} steps · LJSpeech held-out</p>
<h1>ELF-TTS Samples</h1>
<p class="lede">Four tracks per clip, same utterance each time. The two reference tracks bound what is
achievable; the two model tracks separate the flow from the discretizer.</p>
<dl class="spec">
<div><dt>Model</dt><dd>{esc(info['model'])}</dd></div>
<div><dt>Steps</dt><dd>{info['steps']:,}</dd></div>
<div><dt>Weights</dt><dd>{esc(info['weights'])}</dd></div>
<div><dt>NFE</dt><dd>{info['nfe']}</dd></div>
<div><dt>CFG</dt><dd>{info['cfg_scale']}</dd></div>
<div><dt>Prompt</dt><dd>{info['prompt_frac']:.0%}</dd></div>
<div><dt>Code acc</dt><dd>{info['code_acc_gen']*100:.2f}%</dd></div>
<div><dt>TF code acc</dt><dd>{info['code_acc_tf']*100:.2f}%</dd></div>
</dl>
<div class="guide">
<div><b>Original</b> — LJSpeech ground truth. The target.</div>
<div><b>Codec ceiling</b> — the ground truth passed through the frozen codec and back. Any
difference from Original is the tokenizer's loss, not the model's. <b>This is the upper bound.</b></div>
<div><b>Prompt + generated</b> — with a prompt &gt; 0 the opening frames are held at the true
latents throughout sampling, so they decode as ground truth and sound like real speech. That
opening is <b>not</b> the model's output.</div>
<div><b>Generated only</b> — the frames after the prompt, nothing else. <b>This is the system.</b></div>
<div><b>Teacher-forced</b> — the true x₀ handed to the CE head, skipping the flow entirely. If this
sounds close to the ceiling while Generated does not, the flow is the broken part, not the discretizer.</div>
</div>
<div class="warn">Generated code accuracy is <b>{info['code_acc_gen']*100:.2f}%</b> against a
1/4096 = 0.024% floor, so the Generated track is expected to be unintelligible at this checkpoint —
listen to it to confirm what the metric says, and compare Teacher-forced against the ceiling to hear
what the discretizer alone can do.</div>
</header>
{''.join(parts)}
</div></body></html>"""
    with open(path, "w") as f:
        f.write(page)


if __name__ == "__main__":
    main()
