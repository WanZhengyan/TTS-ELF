#!/usr/bin/env python
"""Side-by-side audio from the two architectures, on the same clips.

  ELF-style   flow -> 512-d codec latent -> ELF's decode head (shares the
              backbone with the flow head) -> code -> frozen WavTokenizer
              decoder -> waveform.
              Latent source: a LEARNED encoder, 8.30 M params, pretrained on
              80,000 hours.

  Mel-style   flow -> 100-d log-mel -> frozen Vocos -> waveform. No intermediate
              discretization at all; ELF's decode head is switched off.
              Latent source: NO encoder -- STFT and a fixed mel filterbank,
              zero learned parameters.

Both caches were built from the same sorted, [1,10]s-filtered metadata, so record
i is the same clip in both and the two test splits line up.

The two runs are NOT length-matched: codec used max_length 400 frames = 10.0 s
with no truncation, mel used 384 frames = 4.1 s with 84 % of clips truncated.
That is reported rather than hidden, because it inflates both of the mel
path's WER figures.
"""
import argparse, base64, io, json, os, pickle, sys, html

import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "third_party", "WavTokenizer"))
sys.path.insert(0, HERE)
from train import ELFTTS, LJCache, make_collate, Cfg  # noqa: E402
from sample_mel import rollout_mel, wer_pair  # noqa: E402
from train import sample as elf_sample  # noqa: E402
from decoder.pretrained import WavTokenizer  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def log(*a):
    print(*a, flush=True)


def b64(w, sr=24000):
    buf = io.BytesIO()
    torchaudio.save(buf, w.cpu(), sr, format="wav", encoding="PCM_S", bits_per_sample=16)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec_ckpt_run", default=os.path.join(HERE, "runs", "d_xs_dp02", "ckpt.pt"))
    ap.add_argument("--mel_ckpt_run", default=os.path.join(HERE, "runs", "h_mel_xs", "ckpt.pt"))
    ap.add_argument("--lj_root", default=os.path.join(HERE, "data", "LJSpeech-1.1"))
    ap.add_argument("--codec_cfg", default=os.path.join(
        HERE, "third_party", "WavTokenizer", "configs",
        "wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"))
    ap.add_argument("--codec_w", default=os.path.join(HERE, "ckpt", "wavtokenizer_large_unify_600_24k.ckpt"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--n_wav", type=int, default=6)
    ap.add_argument("--page", default=os.path.join(HERE, "compare_arch.html"))
    ap.add_argument("--out", default=os.path.join(HERE, "out", "compare_arch.json"))
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- the two trained models
    ck_c = torch.load(args.codec_ckpt_run, map_location="cpu", weights_only=False)
    ck_m = torch.load(args.mel_ckpt_run, map_location="cpu", weights_only=False)
    ta_c, ta_m = ck_c["args"], ck_m["args"]
    cfg_c = Cfg(**{k: v for k, v in ck_c["cfg"].items() if k in Cfg.__dataclass_fields__})
    cfg_m = Cfg(**{k: v for k, v in ck_m["cfg"].items() if k in Cfg.__dataclass_fields__})

    ds_c = LJCache(os.path.join(HERE, "cache"), "test", seed=ta_c["seed"])
    ds_m = LJCache(os.path.join(HERE, "cache_mel"), "test", seed=ta_m["seed"])
    meta_c, meta_m = ds_c.meta, ds_m.meta

    m_c = ELFTTS(ta_c["model"], meta_c["char_vocab_size"], meta_c["audio_vocab_size"],
                 dim=512, max_length=ta_c["max_length"]).to(dev)
    m_c.load_state_dict(ck_c["ema"]); m_c.eval()
    m_m = ELFTTS(ta_m["model"], meta_m["char_vocab_size"], meta_m["audio_vocab_size"],
                 dim=meta_m["dim"], max_length=ta_m["max_length"]).to(dev)
    m_m.load_state_dict(ck_m["ema"]); m_m.eval()
    log(f"ELF-style : {ta_c['model']} @ {ta_c['steps']} steps, codec latent, decode head trained")
    log(f"Mel-style : {ta_m['model']} @ {ta_m['steps']} steps, log-mel, decode head off")

    # ---- the two frozen decoders
    wt = WavTokenizer.from_pretrained0802(args.codec_cfg, args.codec_w).to(dev).eval()
    vq = wt.feature_extractor.encodec.quantizer.vq.layers[0]
    bw = torch.tensor([0], device=dev)
    from vocos import Vocos
    voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(dev).eval()

    from transformers import pipeline
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3",
                   device=0 if dev.type == "cuda" else -1,
                   torch_dtype=torch.float16 if dev.type == "cuda" else torch.float32)
    norm = EnglishTextNormalizer({})
    try:
        um = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                            trust_repo=True).to(dev).eval()
    except Exception as e:
        log(f"UTMOS unavailable: {e}"); um = None

    id2text = {}
    with open(os.path.join(args.lj_root, "metadata.csv"), encoding="utf-8") as f:
        for line in f:
            p = line.strip().split("|")
            if len(p) >= 3:
                id2text[p[0]] = p[2]

    # the two splits line up, but intersect by id anyway so this cannot silently drift
    ids_c = [meta_c["records"][i]["id"] for i in ds_c.ids]
    ids_m = [meta_m["records"][i]["id"] for i in ds_m.ids]
    common = [x for x in ids_c if x in set(ids_m)][:args.n]
    log(f"codec test {len(ids_c)} | mel test {len(ids_m)} | common {len(common)}, using {len(common)}")
    pos_c = {meta_c["records"][i]["id"]: j for j, i in enumerate(ds_c.ids)}
    pos_m = {meta_m["records"][i]["id"]: j for j, i in enumerate(ds_m.ids)}

    coll_c = make_collate(ta_c["max_length"], (0.0, 0.0), ta_c["seed"], fixed_prompt=0.0, dim=512)
    coll_m = make_collate(ta_m["max_length"], (0.0, 0.0), ta_m["seed"], fixed_prompt=0.0,
                          dim=meta_m["dim"])
    rs = torchaudio.transforms.Resample(22050, 24000).to(dev)

    tracks = ["orig", "elf_ceiling", "elf_gen", "mel_ceiling", "mel_gen"]
    errs = {t: 0 for t in tracks}; lens = {t: 0 for t in tracks}; mos = {t: [] for t in tracks}
    rows = []

    for n, uid in enumerate(common):
        ref = norm(id2text.get(uid, ""))
        if not ref:
            continue
        bc = coll_c([ds_c[pos_c[uid]]])
        bm = coll_m([ds_m[pos_m[uid]]])
        Tc, Tm = int(bc["T"][0]), int(bm["T"][0])

        pred_c, _, _ = elf_sample(m_c, bc, cfg_c, dev, n_steps=32, cfg_scale=2.0,
                                  schedule="logit_normal", sampler="sde", sde_gamma=1.0)
        code_c = bc["code"].to(dev)
        mel_g = rollout_mel(m_m, bm, cfg_m, dev, n_steps=32, cfg_scale=2.0,
                            sampler="sde", sde_gamma=1.0)
        mel_t = bm["latent"].to(dev)

        w, _ = torchaudio.load(os.path.join(args.lj_root, "wavs", uid + ".wav"))
        g = rs(w.to(dev)); g = g.mean(0, keepdim=True) if g.shape[0] > 1 else g
        waves = {
            "orig": g,
            "elf_ceiling": wt.decode(vq.decode(code_c[0:1, :Tc]), bandwidth_id=bw),
            "elf_gen": wt.decode(vq.decode(pred_c[0:1, :Tc]), bandwidth_id=bw),
            "mel_ceiling": voc.decode(mel_t[0:1, :Tm].transpose(1, 2)),
            "mel_gen": voc.decode(mel_g[0:1, :Tm].transpose(1, 2)),
        }
        for t in tracks:
            a = waves[t].squeeze(0).float().clamp(-1, 1)
            o = asr({"sampling_rate": 24000, "raw": a.cpu().numpy()},
                    generate_kwargs={"language": "english"})
            e, L = wer_pair(ref, norm(o["text"]))
            errs[t] += e; lens[t] += L
            if um is not None:
                mos[t].append(float(um(a.unsqueeze(0), 24000).item()))
        if len(rows) < args.n_wav:
            rows.append({"id": uid, "text": id2text.get(uid, ""),
                         "dur_elf": Tc / meta_c["frame_rate"],
                         "dur_mel": Tm / meta_m["frame_rate"],
                         "b64": {t: b64(waves[t].squeeze(0).unsqueeze(0)) for t in tracks}})
        if (n + 1) % 10 == 0:
            log(f"  {n+1}/{len(common)}  " + "  ".join(
                f"{t} {100*errs[t]/max(lens[t],1):.1f}" for t in tracks))

    res = {"n": len(common), "tracks": {}}
    log("")
    log(f"{'track':<14} {'WER %':>8} {'UTMOS':>8}")
    for t in tracks:
        wv = 100 * errs[t] / max(lens[t], 1)
        uv = float(np.mean(mos[t])) if mos[t] else float("nan")
        log(f"{t:<14} {wv:>8.2f} {uv:>8.3f}")
        res["tracks"][t] = {"wer": wv, "utmos": uv}
    res["note"] = ("not length-matched: ELF-style ran full-length clips, mel-style "
                   "truncates to 4.1 s, which inflates both mel figures")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)

    build_page(rows, res, ta_c, ta_m, args.page)
    log(f"\nwrote {args.out} and {args.page}")


LBL = [
    ("orig", "Original recording", "the target", "ref"),
    ("elf_ceiling", "ELF-style ceiling", "original → codec → back. 40 codes/s, 800× compression.", "ref"),
    ("elf_gen", "ELF-style generated", "flow → codec latent → ELF's trained decode head → code → codec decoder", "elf"),
    ("mel_ceiling", "Mel-style ceiling", "original → log-mel → Vocos. No encoder, no discretization.", "ref"),
    ("mel_gen", "Mel-style generated", "flow → log-mel → frozen Vocos. Decode head switched off.", "mel"),
]


def build_page(rows, res, ta_c, ta_m, path):
    esc = html.escape
    t = res["tracks"]
    arts = []
    for r in rows:
        tr = "".join(
            f'<div class="tk {k4}"><div><b>{esc(l)}</b><span>{esc(d)}</span></div>'
            f'<audio controls preload="none" src="data:audio/wav;base64,{r["b64"][k]}"></audio></div>'
            for k, l, d, k4 in LBL)
        arts.append(
            f'<article><header><h3>{esc(r["id"])}</h3>'
            f'<span>ELF-style {r["dur_elf"]:.1f} s · Mel-style {r["dur_mel"]:.1f} s</span>'
            f'</header><p>{esc(r["text"])}</p>{tr}</article>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Two Architectures Compared</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:ital@1&family=IBM+Plex+Mono&display=swap">
<style>
*{{box-sizing:border-box}}
:root{{color-scheme:light;--g:#f6f7f9;--s:#fff;--s2:#eceff4;--i:#171b22;--i2:#4a5260;--i3:#79818f;
--r:#d7dbe3;--r2:#b7bfcb;--elf:#a85d08;--elf-bg:#f8eddd;--mel:#0d7d72;--mel-bg:#e2f2f0;--warn:#a0262b}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{color-scheme:dark;--g:#0e1116;--s:#161b22;
--s2:#1d232b;--i:#e7eaef;--i2:#a9b1be;--i3:#7d8591;--r:#2a313b;--r2:#3c4550;--elf:#e39c4c;
--elf-bg:#33240f;--mel:#45bdad;--mel-bg:#0f2c2a;--warn:#e4807f}}}}
:root[data-theme=dark]{{color-scheme:dark;--g:#0e1116;--s:#161b22;--s2:#1d232b;--i:#e7eaef;
--i2:#a9b1be;--i3:#7d8591;--r:#2a313b;--r2:#3c4550;--elf:#e39c4c;--elf-bg:#33240f;
--mel:#45bdad;--mel-bg:#0f2c2a;--warn:#e4807f}}
html,body{{margin:0}}body{{background:var(--g);color:var(--i);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:15px;line-height:1.6}}
.w{{max-width:880px;margin:0 auto;padding-left:18px;padding-right:18px;padding-block:0 64px}}
h1{{font-size:clamp(25px,5vw,36px);margin:36px 0 12px;font-weight:600;letter-spacing:-.02em}}
.lede{{color:var(--i2);max-width:62ch;margin:0 0 18px}}
h2{{font-size:18px;margin:30px 0 10px;font-weight:600;padding-bottom:7px;border-bottom:1px solid var(--r2)}}
p{{margin:0 0 13px;max-width:70ch}}
code{{font-family:"IBM Plex Mono",monospace;font-size:.875em;background:var(--s2);padding:.1em .34em;border-radius:2px}}
.tscroll{{overflow-x:auto;border:1px solid var(--r);background:var(--s);margin:0 0 18px}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;min-width:540px}}
th,td{{text-align:left;padding:8px 11px;border-bottom:1px solid var(--r);vertical-align:top}}
thead th{{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.06em;
text-transform:uppercase;color:var(--i3);background:var(--s2);font-weight:600;white-space:nowrap}}
tbody tr:last-child td{{border-bottom:none}}
td.n{{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;white-space:nowrap}}
tr.e{{background:var(--elf-bg)}}tr.m{{background:var(--mel-bg)}}
.warn{{border-left:2px solid var(--warn);padding:11px 14px;background:var(--s2);border-radius:0 3px 3px 0;font-size:13.5px;margin:0 0 16px}}
article{{border:1px solid var(--r);background:var(--s);margin:18px 0 0}}
article>header{{padding:11px 14px;border-bottom:1px solid var(--r);background:var(--s2);display:flex;gap:14px;align-items:baseline;flex-wrap:wrap}}
h3{{margin:0;font-family:"IBM Plex Mono",monospace;font-size:13.5px}}
header span{{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--i3)}}
article p{{margin:0;padding:11px 14px;font-family:"IBM Plex Serif",Georgia,serif;font-style:italic;color:var(--i2);border-bottom:1px solid var(--r)}}
.tk{{padding:9px 14px;border-bottom:1px solid var(--r);display:grid;grid-template-columns:minmax(0,268px) 1fr;gap:6px 16px;align-items:center}}
.tk:last-child{{border-bottom:none}}
.tk div{{display:flex;flex-direction:column}}
.tk b{{font-size:13.5px}}.tk span{{font-size:11.5px;color:var(--i3);line-height:1.4}}
.tk.elf b{{color:var(--elf)}}.tk.mel b{{color:var(--mel)}}
audio{{width:100%;height:34px}}
@media(max-width:640px){{.tk{{grid-template-columns:1fr}}}}
</style></head><body><div class="w">
<h1>Two Architectures Compared</h1>
<p class="lede">Same held-out clips, same text-only conditioning, same sampler. What differs is what
the flow generates and what turns it into sound.</p>

<h2>The two designs</h2>
<div class="tscroll"><table>
<thead><tr><th></th><th>ELF-style</th><th>Mel-style</th></tr></thead>
<tbody>
<tr><td>Flow generates</td><td>512-d codec latent, 40 Hz</td><td>100-d log-mel, 93.75 Hz</td></tr>
<tr><td><b>Latent comes from</b></td><td><b>a learned encoder</b> — SEANet, 8.30 M params, pretrained on 80,000 h</td><td><b>no encoder</b> — STFT and a fixed mel filterbank, 0 learned params</td></tr>
<tr><td>Intermediate discretization</td><td>ELF's decode head, <b>trained</b>, sharing the 10.65 M backbone with the flow head</td><td><b>none</b> — switched off, nothing discrete to decode into</td></tr>
<tr><td>Final decoder</td><td>WavTokenizer decoder, frozen</td><td>Vocos 13.53 M, frozen, shares no parameters with our model</td></tr>
<tr><td>Effective positions to get right</td><td class="n">204</td><td class="n">18</td></tr>
<tr><td>Clip length evaluated</td><td class="n">full, up to 10.0 s</td><td class="n">truncated to 4.1 s</td></tr>
</tbody></table></div>

<h2>Measured, {res['n']} held-out clips</h2>
<div class="tscroll"><table>
<thead><tr><th>Track</th><th>WER&nbsp;%</th><th>UTMOS</th></tr></thead>
<tbody>
<tr><td>Original recording</td><td class="n">{t['orig']['wer']:.2f}</td><td class="n">{t['orig']['utmos']:.3f}</td></tr>
<tr class="e"><td>ELF-style ceiling</td><td class="n">{t['elf_ceiling']['wer']:.2f}</td><td class="n">{t['elf_ceiling']['utmos']:.3f}</td></tr>
<tr class="e"><td><b>ELF-style generated</b></td><td class="n"><b>{t['elf_gen']['wer']:.2f}</b></td><td class="n"><b>{t['elf_gen']['utmos']:.3f}</b></td></tr>
<tr class="m"><td>Mel-style ceiling</td><td class="n">{t['mel_ceiling']['wer']:.2f}</td><td class="n">{t['mel_ceiling']['utmos']:.3f}</td></tr>
<tr class="m"><td><b>Mel-style generated</b></td><td class="n"><b>{t['mel_gen']['wer']:.2f}</b></td><td class="n"><b>{t['mel_gen']['utmos']:.3f}</b></td></tr>
</tbody></table></div>
<p class="warn"><b>Read each generated track against its own ceiling, not against the other
architecture.</b> The two runs are not length-matched: ELF-style was evaluated on full clips, mel-style
on 4.1-second truncations while the WER reference stays the whole transcript, which inflates both mel
numbers by roughly 33 points of missing words. The honest quantity is
<code>generated − ceiling</code> within each column.</p>

<h2>Listen</h2>
<p>Each clip has both ceilings and both generated tracks. The ceiling tells you what its own
representation can carry at best; the generated track tells you how close the flow got.</p>
{''.join(arts)}
</div></body></html>"""
    open(path, "w").write(page)


if __name__ == "__main__":
    main()
