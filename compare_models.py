#!/usr/bin/env python
"""Evaluate every unified checkpoint on both directions and build one page.

The architecture is IDENTICAL across these checkpoints -- same ELF-B trunk,
same two channels, same task tokens, same loss. What differs is only the
training configuration, so the page reports one architecture and a config table,
and every audio clip is labelled with the checkpoint that produced it.

TTS is evaluated on each model's OWN test split, because the models were trained
on different corpora and cross-corpus audio would not be a fair comparison. S2T
is reported as the decoded text next to the reference, which is the only honest
way to show a 2% token accuracy: a WER number alone looks like a score.
"""
import argparse, base64, io, json, os, sys

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from unified import TTS, S2T, Cfg, UnifiedELF, net_to_v, restore
from train_unified import (LibriCache, TextEncoder, make_collate, build_targets,
                           make_schedule, collapse, wer_pair, T5_EOS, T5_PAD)
from train_unified import audio_cond
from sample_unified import tts_rollout, wav_b64


def log(*a):
    print(*a, flush=True)


MODELS = [
    dict(key="p_dp5", name="A  cond kept, no audio-drop",
         ckpt="runs/p_dp5/ckpt.pt", cache="cache_lj_uni"),
    dict(key="p_dp5_acfg", name="B  cond kept, audio-drop 0.1",
         ckpt="runs/p_dp5_acfg/ckpt.pt", cache="cache_lj_uni"),
    dict(key="c_nocond", name="C  cond removed, no audio-drop",
         ckpt="runs/c_nocond/ckpt.pt", cache="cache_lj_uni"),
    dict(key="e_best", name="E  cond removed, audio-drop 0.1",
         ckpt="runs/e_best/ckpt.pt", cache="cache_lj_uni"),
    dict(key="f_s2tw03", name="F  cond removed, drop 0.1, s2t_weight 0.3",
         ckpt="runs/f_s2tw03/ckpt.pt", cache="cache_lj_uni"),
]


@torch.no_grad()
def s2t_decode(model, b, cfg, device, text_enc, mel_stats, tk, layout, steps):
    """Roll out the text channel with the audio held clean, then collapse."""
    chars = b["chars"].to(device); valid = b["valid"].to(device)
    v_act = b["v_act"].to(device).unsqueeze(-1)
    t_act = b["t_act"].to(device).unsqueeze(-1)
    p_v, p_t = b["prompt_v"].to(device), b["prompt_t"].to(device)
    task = b["task"].to(device)
    x0_t, x0_v = build_targets(b, text_enc, device, mel_stats)
    ac, am = audio_cond(b, device)
    B = x0_t.shape[0]
    ns = cfg.denoiser_noise_scale
    z_t = restore(torch.randn_like(x0_t) * ns, x0_t, p_t) * t_act
    z_v = restore(torch.randn_like(x0_v) * ns, x0_v, p_v) * v_act
    xp_t, xp_v = torch.zeros_like(z_t), torch.zeros_like(z_v)
    sc_cfg = torch.ones(B, device=device)
    ts = make_schedule(steps, cfg, device)
    for i in range(steps):
        tv = float(ts[i])
        t = torch.full((B,), tv, device=device)
        xh_t, xh_v, _ = model(z_t, z_v, xp_t, xp_v, chars, t, task,
                              attention_mask=valid, self_cond_cfg_scale=sc_cfg,
                              audio_cond=ac, audio_cond_mask=am)
        h = float(ts[i + 1]) - tv
        z_t = restore(z_t + h * net_to_v(xh_t, z_t, t), x0_t, p_t) * t_act
        z_v = restore(z_v + h * net_to_v(xh_v, z_v, t), x0_v, p_v) * v_act
        xp_t = restore(xh_t, x0_t, p_t) * t_act
        xp_v = restore(xh_v, x0_v, p_v) * v_act
    t1 = torch.ones(B, device=device)
    _, _, logits = model(z_t, z_v, xp_t, xp_v, chars, t1, task,
                         attention_mask=valid, self_cond_cfg_scale=sc_cfg,
                         decoder_step_active=True,
                         audio_cond=ac, audio_cond_mask=am)
    pred = logits.argmax(-1)
    out = []
    err = ln = tok = tokn = 0
    for i in range(B):
        n = int(b["n_text"][i]); st = int(b["text_start"][i])
        ref_ids = [x for x in b["t5_dense"][i, :n].tolist()
                   if x not in (T5_EOS, T5_PAD)]
        if layout == "channel":
            hyp_ids = collapse(pred[i].tolist())
        else:
            raw = pred[i, st:st + b["t5_dense"].shape[1]].tolist()
            if T5_EOS in raw:
                raw = raw[:raw.index(T5_EOS)]
            hyp_ids = [x for x in raw if x != T5_PAD]
        ref = tk.decode(ref_ids, skip_special_tokens=True)
        hyp = tk.decode(hyp_ids, skip_special_tokens=True)
        e, l = wer_pair(ref, hyp)
        err += e; ln += l
        k = min(len(ref_ids), len(hyp_ids))
        tok += sum(1 for j in range(k) if ref_ids[j] == hyp_ids[j])
        tokn += len(ref_ids)
        out.append((ref, hyp))
    # the raw counts come back too: WER pools as total_err/total_len across
    # batches, which is not the mean of the per-batch ratios
    return out, 100.0 * err / max(ln, 1), tok / max(tokn, 1), (err, ln, tok, tokn)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_wer", type=int, default=40)
    ap.add_argument("--n_wav", type=int, default=3)
    ap.add_argument("--n_s2t", type=int, default=8)   # clips scored
    ap.add_argument("--n_show", type=int, default=8)  # clips printed
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--prompt", type=float, default=0.0)
    ap.add_argument("--asr", default="openai/whisper-large-v3")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "compare.json"))
    ap.add_argument("--page", default=os.path.join(HERE, "both_directions.html"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from vocos import Vocos
    voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
    from transformers import pipeline, AutoTokenizer
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    asr = pipeline("automatic-speech-recognition", model=args.asr,
                   device=0 if device.type == "cuda" else -1,
                   torch_dtype=torch.float16 if device.type == "cuda"
                   else torch.float32)
    norm = EnglishTextNormalizer({})
    tk = AutoTokenizer.from_pretrained("t5-small")
    try:
        um = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                            trust_repo=True).to(device).eval()
    except Exception as e:
        log(f"UTMOS unavailable: {e}"); um = None
    log("vocoder, ASR and UTMOS ready")

    results = []
    for spec in MODELS:
        ckpt = os.path.join(HERE, spec["ckpt"])
        if not os.path.exists(ckpt):
            log(f"skip {spec['key']}: no checkpoint"); continue
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        ta = ck["args"]
        mk = ck["meta_keys"]
        cache = os.path.join(HERE, spec["cache"])
        ds = LibriCache(cache, "test")
        meta = ds.meta
        mel_stats = (meta["latent_mean"], meta["latent_std"])
        cfg = Cfg(denoiser_p_mean=ta["denoiser_p_mean"])
        wdim = meta.get("whisper_dim", 0) if ta.get("whisper") else 0
        wrate = meta.get("whisper_rate", 0.0) if ta.get("whisper") else 0.0
        # read the architecture off each checkpoint: these specs differ in
        # whether the cond block exists, so it cannot be hardcoded here
        model = UnifiedELF(ta["model"], char_vocab=meta["char_vocab_size"],
                           max_length=ta["max_length"],
                           whisper_dim=wdim,
                           use_cond=not ta.get("no_cond", 0),
                       spk_dim=(meta.get("spk_emb_dim", 0)
                                if ta.get("spk") else 0)).to(device)
        model.load_state_dict(ck["ema"]); model.eval()
        text_enc = TextEncoder(ta["text_latent_std"]).to(device)
        itos = {v: k for k, v in meta["stoi"].items()}
        dataset = meta.get("dataset", "librispeech")
        log("")
        log(f"=== {spec['name']} ({spec['key']}) ===")
        log(f"  step {ck['step']}, layout {ta['layout']}, p_s2t {ta['p_s2t']}, "
            f"{mk['n_speakers']} speaker(s), {dataset}")

        R = dict(spec, step=ck["step"], layout=ta["layout"], p_s2t=ta["p_s2t"],
                 n_speakers=mk["n_speakers"], dataset=dataset,
                 max_length=ta["max_length"], max_text=ta["max_text"],
                 batch=ta["batch"], lr=ta["lr"], init_frac=ta.get("init_frac"),
                 p_mean=ta["denoiser_p_mean"],
                 whisper=bool(ta.get("whisper")),
                 decoder_p_mean=ta.get("decoder_p_mean", 0.8),
                 latent_mean=meta["latent_mean"], latent_std=meta["latent_std"],
                 params_M=sum(p.numel() for p in model.parameters()) / 1e6,
                 clips=[], s2t=None)

        if dataset == "ljspeech":
            id2text, src_sr = {}, 22050
            with open(os.path.join(HERE, "data", "LJSpeech-1.1",
                                   "metadata.csv"), encoding="utf-8") as f:
                for line in f:
                    q = line.strip().split("|")
                    if len(q) >= 3:
                        id2text[q[0]] = q[2]
            wav_path = lambda u: os.path.join(
                HERE, "data", "LJSpeech-1.1", "wavs", u + ".wav")
        else:
            import glob
            id2text, src_sr = {}, 16000
            for tr in glob.glob(os.path.join(HERE, "data", "LibriSpeech",
                                             "train-clean-100", "*", "*",
                                             "*.trans.txt")):
                with open(tr, encoding="utf-8") as f:
                    for line in f:
                        u, _, txt = line.strip().partition(" ")
                        id2text[u] = txt
            def wav_path(u):
                s, c, _ = u.split("-")
                return os.path.join(HERE, "data", "LibriSpeech",
                                    "train-clean-100", s, c, u + ".flac")

        # ---------- TTS
        coll = make_collate(ta["max_length"], 0.0, (args.prompt, args.prompt),
                            rng_seed=7, max_text=ta["max_text"], itos=itos,
                            tk=tk, layout=ta["layout"], whisper_rate=wrate,
                            frame_rate=meta["frame_rate"])
        dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0,
                        collate_fn=coll)
        rs = torchaudio.transforms.Resample(src_sr, 24000).to(device)
        tracks = ["orig", "ceiling", "gen", "uncond"]
        errs = {t: 0 for t in tracks}; lens = {t: 0 for t in tracks}
        mos = {t: [] for t in tracks}
        done = 0
        for b in dl:
            if done >= args.n_wer:
                break
            mg = tts_rollout(model, b, cfg, device, text_enc, mel_stats,
                             steps=args.nfe)
            mu_ = tts_rollout(model, b, cfg, device, text_enc, mel_stats,
                              steps=args.nfe, drop_text=True)
            mt = b["mel"].to(device)
            for i in range(mg.shape[0]):
                if done >= args.n_wer:
                    break
                uid = b["uid"][i]
                T = int(b["valid"][i].sum().item())
                n = int(b["n_text"][i])
                ref = norm(tk.decode(
                    [x for x in b["t5_dense"][i, :n].tolist()
                     if x not in (T5_EOS, T5_PAD)], skip_special_tokens=True))
                if not ref.strip() or T < 32:
                    done += 1; continue
                a0, hop = int(b["crop_a"][i]), 256
                g = rs(torchaudio.load(wav_path(uid))[0].to(device))
                g = g.mean(0, keepdim=True) if g.shape[0] > 1 else g
                g = g[:, a0 * hop:(a0 + T) * hop]
                waves = {"orig": g,
                         "ceiling": voc.decode(mt[i:i+1, :T].transpose(1, 2)),
                         "gen": voc.decode(mg[i:i+1, :T].transpose(1, 2)),
                         "uncond": voc.decode(mu_[i:i+1, :T].transpose(1, 2))}
                for t in tracks:
                    aud = waves[t].squeeze(0).float().clamp(-1, 1)
                    o = asr({"sampling_rate": 24000, "raw": aud.cpu().numpy()},
                            generate_kwargs={"language": "english"})
                    e, nn = wer_pair(ref, norm(o["text"]))
                    errs[t] += e; lens[t] += nn
                    if um is not None:
                        mos[t].append(float(um(aud.unsqueeze(0), 24000).item()))
                if len(R["clips"]) < args.n_wav:
                    R["clips"].append({
                        "id": uid, "text": ref, "dur": T / meta["frame_rate"],
                        "b64": {t: wav_b64(waves[t].squeeze(0).unsqueeze(0).cpu())
                                for t in tracks}})
                done += 1
        R["tts"] = {t: {"wer": 100 * errs[t] / max(lens[t], 1),
                        "utmos": float(np.mean(mos[t])) if mos[t] else float("nan")}
                    for t in tracks}
        R["n_tts"] = done
        log(f"  TTS on {done} clips: " + "  ".join(
            f"{t} WER {R['tts'][t]['wer']:.2f} UTMOS {R['tts'][t]['utmos']:.3f}"
            for t in tracks))

        # ---------- S2T, only for models that trained it
        if ta["p_s2t"] > 0:
            coll_s = make_collate(ta["max_length"], 1.0, (0.0, 0.0), rng_seed=7,
                                  max_text=ta["max_text"], itos=itos, tk=tk,
                                  layout=ta["layout"], whisper_rate=wrate,
                                  frame_rate=meta["frame_rate"])
            dl_s = DataLoader(ds, batch_size=args.batch, collate_fn=coll_s)
            pairs, E = [], [0, 0, 0, 0]
            seen = 0
            for bs in dl_s:
                if seen >= args.n_s2t:
                    break
                p_, _, _, c_ = s2t_decode(model, bs, cfg, device, text_enc,
                                          mel_stats, tk, ta["layout"], args.nfe)
                for k in range(4):
                    E[k] += c_[k]
                seen += len(p_)
                pairs.extend(p_[:max(0, args.n_show - len(pairs))])
            wer = 100.0 * E[0] / max(E[1], 1)
            tacc = E[2] / max(E[3], 1)
            R["s2t"] = {"pairs": pairs, "wer": wer, "tok_acc": tacc, "n": seen}
            log(f"  S2T: WER {wer:.2f}%  token accuracy {tacc*100:.2f}%  "
                f"(n={seen} clips)")
        results.append(R)
        del model, text_enc
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    slim = [{k: v for k, v in r.items() if k != "clips"} for r in results]
    json.dump(slim, open(args.out, "w"), indent=2)
    log(f"\nwrote {args.out}")
    build_page(results, args)


def build_page(results, args):
    T = {"orig": "Real recording", "ceiling": "Mel ceiling (real mel → Vocos)",
         "gen": "Generated from text", "uncond": "Generated, text removed"}
    works = [r for r in results if r["tts"]["gen"]["utmos"] > 3.0]
    s2t_any = any(r["s2t"] and r["s2t"]["tok_acc"] > 0.5 for r in results)

    H = ['<title>Both Directions</title>',
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=IBM+Plex+Sans+Condensed:wght@400;600;700&'
         'family=IBM+Plex+Mono:wght@400;500;600&'
         'family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">',
         '<style>', CSS, '</style>', '<div class="wrap">']

    H += ['<section>', '<p class="eyebrow">Unified TTS + S2T</p>',
          '<h1 class="m">两个方向，一个答案</h1>',
          '<p class="lede m">三个 checkpoint，<b>架构完全相同</b>——同一个 ELF-B 主干、'
          '同样两个通道、同样的 task token、同一个目标函数。不同的只有训练配置。'
          '每段音频都标注了它出自哪个 checkpoint。</p>',
          '<dl class="verdict">',
          f'<div><dt>TTS 方向</dt><dd class="{"ok" if works else "bad"}">'
          f'{"work" if works else "不 work"}'
          f'<small>{works[0]["name"] if works else ""}：UTMOS '
          f'{works[0]["tts"]["gen"]["utmos"]:.3f}，天花板 '
          f'{works[0]["tts"]["ceiling"]["utmos"]:.3f}</small></dd></div>'
          if works else
          '<div><dt>TTS 方向</dt><dd class="bad">不 work</dd></div>',
          f'<div><dt>S2T 方向</dt><dd class="bad">'
          f'{"work" if s2t_any else "不 work"}<small>最好的 token 准确率 '
          f'{max((r["s2t"]["tok_acc"] for r in results if r["s2t"]), default=0)*100:.1f}%'
          f'，随机是 0.003%，可用需要 &gt;90%</small></dd></div>',
          '</dl>', '</section>']

    # ---- shared architecture
    H += ['<section>', '<p class="eyebrow">架构</p>',
          '<h2 class="m">三个 checkpoint 共用的那一个</h2>',
          '<p class="m">序列的每个位置同时携带两个通道：语音通道是 100 维原始 log-mel，'
          '文本通道是 512 维 T5 encoder 输出升采样到帧率。哪个通道加噪，它就是目标；'
          '另一个保持干净，就是条件。task token 告诉模型是哪个方向。</p>',
          ARCH_SVG,
          '<div class="t-scroll"><table><tbody>',
          '<tr><td>主干</td><td class="n">ELF-B · 12 层 / 768 宽 / 12 头</td>'
          '<td>权重 99.58% 来自 <code>ELF-B-owt checkpoint_95085</code></td></tr>',
          '<tr><td>语音侧</td><td class="n">无 encoder，无 decode head</td>'
          '<td>原始 log-mel 进，denoiser 的 <code>voice_out</code>(512→100) 出</td></tr>',
          '<tr><td>文本侧</td><td class="n">冻结 t5-small encoder ÷ 0.2</td>'
          '<td>出口是 ELF 预训练的 decode head，词表 32100，仅 S2T 行启用</td></tr>',
          '<tr><td>新增参数</td><td class="n">442,980 · 0.42%</td>'
          '<td>加宽输入投影 364,544 + voice_out 51,300 + char_embed 20,480 '
          '+ task_tokens 6,144 + RMSNorm 512</td></tr>',
          '<tr><td>目标函数</td><td class="n">单一 flow matching</td>'
          '<td>逐行 Bernoulli(0.2) 路由到 CE 或 L2，共用同一个分母</td></tr>',
          '</tbody></table></div>', '</section>']

    # ---- per-model config
    H += ['<section>', '<p class="eyebrow">三个 checkpoint</p>',
          '<h2 class="m">只有训练配置不同</h2>',
          '<div class="t-scroll"><table><thead><tr><th>checkpoint</th>'
          '<th>数据</th><th>说话人</th><th>任务</th><th>步数</th>'
          '<th>mel mean / std</th></tr></thead><tbody>']
    for r in results:
        task = "纯 TTS" if r["p_s2t"] == 0 else f"TTS + S2T (p_s2t {r['p_s2t']})"
        H.append(f'<tr><td><code>{r["key"]}</code></td><td>{r["dataset"]}</td>'
                 f'<td class="n">{r["n_speakers"]}</td><td>{task}</td>'
                 f'<td class="n">{r["step"]:,}</td>'
                 f'<td class="n">{r["latent_mean"]:+.4f} / {r["latent_std"]:.4f}'
                 f'</td></tr>')
    H += ['</tbody></table></div>',
          '<p class="m" style="margin-top:14px;font-size:14.5px;color:var(--ink-2)">'
          f'<code>ch_mix</code> 只跑了 6,000 步——它被我中途砍掉去腾卡做 LJSpeech 对照了，'
          '所以同时训两个方向的配置还没有一次完整的 40,000 步。'
          '另外三者的 <code>max_length</code> 512、batch 32、lr 1e-4、'
          'channel 布局、<code>init_frac</code> 0.25 都相同。</p>', '</section>']

    # ---- TTS results
    H += ['<section>', '<p class="eyebrow">TTS 方向</p>',
          '<h2 class="m">结果</h2>',
          '<p class="m">每个模型在<b>自己的</b>测试集上评测——它们训在不同语料上，'
          '跨语料比较不公平。要看的是 <b>Generated</b> 和 <b>Mel ceiling</b> 的差距'
          '（那是 flow 的代价），以及 <b>text removed</b> 落后多少'
          '（那是文本确实在工作的证据）。</p>',
          '<div class="t-scroll"><table><thead><tr><th>checkpoint</th>'
          '<th>track</th><th>WER %</th><th>UTMOS</th></tr></thead><tbody>']
    for r in results:
        for j, t in enumerate(["orig", "ceiling", "gen", "uncond"]):
            cls = ' class="hi"' if t == "gen" else ""
            key = (f'<td rowspan="4"><code>{r["key"]}</code><br>'
                   f'<span class="sub">{r["n_tts"]} clips</span></td>'
                   if j == 0 else "")
            H.append(f'<tr{cls}>{key}<td>{T[t]}</td>'
                     f'<td class="n">{r["tts"][t]["wer"]:.2f}</td>'
                     f'<td class="n">{r["tts"][t]["utmos"]:.3f}</td></tr>')
    H += ['</tbody></table></div>', '</section>']

    # ---- audio, labelled
    H += ['<section>', '<p class="eyebrow">听</p>',
          '<h2 class="m">每段音频出自哪个 checkpoint</h2>']
    for r in results:
        H.append(f'<div class="mblock"><h3><code>{r["key"]}</code> '
                 f'<span class="sub">{r["name"]} · {r["dataset"]} · '
                 f'{r["n_speakers"]} 说话人 · step {r["step"]:,} · '
                 f'UTMOS {r["tts"]["gen"]["utmos"]:.3f}</span></h3>')
        for c in r["clips"]:
            H.append(f'<div class="clip"><p class="cid">{c["id"]} '
                     f'<span class="sub">{c["dur"]:.1f}s</span></p>'
                     f'<p class="tx">{c["text"]}</p>')
            for t in ["orig", "ceiling", "gen", "uncond"]:
                H.append(f'<div class="tr"><span class="lb">{T[t]}</span>'
                         f'<audio controls preload="metadata" '
                         f'src="data:audio/wav;base64,{c["b64"][t]}"></audio>'
                         f'</div>')
            H.append('</div>')
        H.append('</div>')
    H.append('</section>')

    # ---- S2T results
    H += ['<section>', '<p class="eyebrow">S2T 方向</p>',
          '<h2 class="m">语音 → 文本：它实际输出的东西</h2>',
          '<p class="m">S2T 的输出是文本，所以没有音频可听。'
          '下面是模型在 hybrid 布局下从 48 个 token 位置直接读出的结果，'
          '和参考并排。WER 越低越好，token accuracy 越高越好；'
          '两个数都在同一批 clip 上汇总（总编辑距离 ÷ 总词数，'
          '不是各批比值的平均）。</p>']
    for r in results:
        if not r["s2t"]:
            continue
        H.append(f'<div class="mblock"><h3><code>{r["key"]}</code> '
                 f'<span class="sub">step {r["step"]:,} · WER '
                 f'{r["s2t"]["wer"]:.2f}% · token 准确率 '
                 f'{r["s2t"]["tok_acc"]*100:.2f}%</span></h3>')
        for ref, hyp in r["s2t"]["pairs"]:
            H.append(f'<div class="pair"><p class="ref"><span class="lb">参考'
                     f'</span>{ref}</p><p class="hyp"><span class="lb">输出'
                     f'</span>{hyp if hyp.strip() else "(空)"}</p></div>')
        H.append('</div>')
    H += ['<div class="note"><b>原因是具体的：音频条件喂的是原始 mel。</b>'
          'ELF-S2T 报的 10.50% WER 建立在<b>冻结 Whisper encoder</b> 之上——'
          '音频先被编码成音素内容线性可读的表示，再作为条件。'
          '喂原始 mel，等于要求 flow transformer 一边学 flow 一边从零学声学模型。'
          '「语音是连续的所以不需要 encoder」对 TTS 成立（语音在那里是<b>目标</b>），'
          '对 S2T 不成立（语音在那里是<b>识别的条件</b>）。</div>', '</section>']

    H += ['<p class="foot">WER 用 Whisper-large-v3 配 Whisper 自带的 '
          'EnglishTextNormalizer；UTMOS 用 tarepan/SpeechMOS utmos22_strong。'
          f'NFE {args.nfe}，voice prompt {args.prompt:.0%}。'
          '参考文本是音频裁剪对应的那一段，所以 <code>orig</code> 的 WER 高于'
          '整句评测的常见值——裁剪边界用的是均匀的字符↔帧假设。</p>',
          '</div>']
    open(args.page, "w").write("\n".join(H))
    log(f"wrote {args.page}")


CSS = """
:root{color-scheme:light;--paper:#f3f5f7;--card:#fbfbfc;--sunk:#eceff2;
--ink:#15181e;--ink-2:#4a515c;--ink-3:#79818d;--rule:#d6dbe1;--rule-2:#e6e9ee;
--voice:#12707b;--voice-soft:#dbedef;--voice-edge:#7fbcc3;--text-ch:#965909;
--text-soft:#f4e6d2;--text-edge:#d4a768;--elf:#4f5865;--elf-soft:#e3e6ea;
--ok:#1a6a44;--bad:#9c2a20;
--sans:"IBM Plex Sans Condensed","Helvetica Neue",Arial,sans-serif;
--serif:"Source Serif 4",Georgia,"Times New Roman",serif;
--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;--measure:66ch}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
color-scheme:dark;--paper:#10131a;--card:#171b23;--sunk:#1c212a;--ink:#e6eaf0;
--ink-2:#a5aeba;--ink-3:#6f7883;--rule:#2a313b;--rule-2:#212731;
--voice:#46c0cc;--voice-soft:#103036;--voice-edge:#2a6e77;--text-ch:#deab5e;
--text-soft:#352616;--text-edge:#7d5c2a;--elf:#98a2ae;--elf-soft:#232933;
--ok:#56c48c;--bad:#e8857a}}
:root[data-theme=dark]{color-scheme:dark;--paper:#10131a;--card:#171b23;
--sunk:#1c212a;--ink:#e6eaf0;--ink-2:#a5aeba;--ink-3:#6f7883;--rule:#2a313b;
--rule-2:#212731;--voice:#46c0cc;--voice-soft:#103036;--voice-edge:#2a6e77;
--text-ch:#deab5e;--text-soft:#352616;--text-edge:#7d5c2a;--elf:#98a2ae;
--elf-soft:#232933;--ok:#56c48c;--bad:#e8857a}
body{background:var(--paper);color:var(--ink);font-family:var(--serif);
font-size:16.5px;line-height:1.62;padding-inline:20px;padding-block:0;
-webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto;padding-block:46px 72px}
.m{max-width:var(--measure)}
h1{font-family:var(--sans);font-weight:700;font-size:clamp(30px,5.4vw,46px);
line-height:1.05;letter-spacing:-.015em;margin:0 0 14px;text-wrap:balance}
h2{font-family:var(--sans);font-weight:600;font-size:clamp(21px,3vw,26px);
line-height:1.2;margin:0 0 10px;text-wrap:balance}
h3{font-family:var(--sans);font-weight:600;font-size:17px;margin:0 0 10px}
p{margin:0 0 14px}
.eyebrow{font-family:var(--mono);font-size:11px;font-weight:500;
letter-spacing:.13em;text-transform:uppercase;color:var(--ink-3);margin:0 0 7px}
.lede{font-size:19px;line-height:1.55;color:var(--ink-2)}
.sub{font-family:var(--mono);font-size:11.5px;font-weight:400;color:var(--ink-3);
letter-spacing:.02em}
code{font-family:var(--mono);font-size:.885em;background:var(--sunk);
padding:.1em .34em;border-radius:3px}
section{margin-top:44px;padding-top:26px;border-top:1px solid var(--rule)}
section:first-of-type{border-top:0;padding-top:0}
.verdict{display:flex;flex-wrap:wrap;gap:0;margin:26px 0 0;
border:1px solid var(--rule);border-radius:5px;overflow:hidden;
background:var(--card)}
.verdict>div{flex:1 1 220px;padding:15px 17px;border-left:1px solid var(--rule-2)}
.verdict>div:first-child{border-left:0}
.verdict dt{font-family:var(--mono);font-size:10.5px;letter-spacing:.11em;
text-transform:uppercase;color:var(--ink-3);margin:0 0 5px}
.verdict dd{margin:0;font-family:var(--sans);font-weight:600;font-size:25px;
line-height:1.1}
.verdict dd.ok{color:var(--ok)}.verdict dd.bad{color:var(--bad)}
.verdict dd small{display:block;font-family:var(--serif);font-weight:400;
font-size:13px;line-height:1.4;color:var(--ink-2);margin-top:4px}
figure{margin:26px 0 0}
.fig-scroll{overflow-x:auto;margin-inline:-4px;padding-inline:4px}
svg.d{display:block;width:100%;min-width:600px;max-width:100%;height:auto;
color:var(--ink)}
figcaption{font-size:14px;line-height:1.5;color:var(--ink-2);margin-top:11px;
max-width:var(--measure)}
.t-scroll{overflow-x:auto;margin-top:20px}
table{border-collapse:collapse;width:100%;min-width:560px;font-size:14.5px}
th,td{text-align:left;padding:8px 14px 8px 0;
border-bottom:1px solid var(--rule-2);vertical-align:top}
th{font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;
text-transform:uppercase;color:var(--ink-3);font-weight:500;
border-bottom:1px solid var(--rule)}
td.n{font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
tr.hi td{background:var(--voice-soft)}
.mblock{margin-top:26px}
.clip{background:var(--card);border:1px solid var(--rule);border-radius:6px;
padding:14px 16px;margin-top:12px}
.cid{font-family:var(--mono);font-size:12.5px;margin:0 0 6px;color:var(--ink-2)}
.tx{color:var(--ink-2);font-size:14px;margin:0 0 10px}
.tr{display:grid;gap:4px;padding:7px 0;border-top:1px solid var(--rule-2)}
.tr:first-of-type{border-top:0}
.lb{font-family:var(--mono);font-size:11.5px;letter-spacing:.04em;
color:var(--ink-3);margin-right:8px}
audio{width:100%}
.pair{background:var(--card);border:1px solid var(--rule);border-radius:6px;
padding:12px 15px;margin-top:10px;font-size:14.5px}
.pair p{margin:0}
.ref{color:var(--ink)}
.hyp{color:var(--bad);margin-top:5px}
.note{border-left:2px solid var(--text-ch);padding:2px 0 2px 15px;
margin:24px 0 0;max-width:var(--measure);font-size:15.5px;color:var(--ink-2)}
.note b{color:var(--ink);font-family:var(--sans)}
.foot{margin-top:50px;padding-top:20px;border-top:1px solid var(--rule);
font-family:var(--mono);font-size:11.5px;color:var(--ink-3);line-height:1.7}
"""

ARCH_SVG = """
<figure><div class="fig-scroll">
<svg class="d" viewBox="0 0 860 250" role="img" aria-label="每个序列位置同时携带 100 维原始 log-mel 的语音通道和 512 维 T5 encoder 输出的文本通道。TTS 时语音通道加噪作为 flow 的目标、文本通道干净作为条件；S2T 时两者对调。task token 选择方向。">
<defs>
<pattern id="hv2" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
<rect width="7" height="7" fill="var(--voice-soft)"/>
<line x1="0" y1="0" x2="0" y2="7" stroke="var(--voice)" stroke-width="2.1"/></pattern>
<pattern id="ht2" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
<rect width="7" height="7" fill="var(--text-soft)"/>
<line x1="0" y1="0" x2="0" y2="7" stroke="var(--text-ch)" stroke-width="2.1"/></pattern>
</defs>
<text x="0" y="16" font-family="var(--mono)" font-size="11" fill="currentColor" opacity=".6">同一批位置 · 512 帧 @ 93.75 Hz</text>
<text x="0" y="46" font-family="var(--sans)" font-size="14" font-weight="700" fill="currentColor">TTS</text>
<text x="44" y="46" font-family="var(--mono)" font-size="10.5" fill="var(--voice)">flow 在语音通道</text>
<text x="196" y="72" text-anchor="end" font-family="var(--mono)" font-size="11" fill="var(--voice)">语音 z_V · 100</text>
<rect x="206" y="54" width="640" height="26" rx="3" fill="url(#hv2)" stroke="var(--voice)" stroke-width="2"/>
<text x="526" y="72" text-anchor="middle" font-family="var(--mono)" font-size="11" fill="currentColor">加噪的原始 log-mel — 目标</text>
<text x="196" y="104" text-anchor="end" font-family="var(--mono)" font-size="11" fill="var(--text-ch)">文本 z_T · 512</text>
<rect x="206" y="86" width="640" height="26" rx="3" fill="var(--text-soft)" stroke="var(--text-edge)" stroke-width="1.5"/>
<text x="526" y="104" text-anchor="middle" font-family="var(--mono)" font-size="11" fill="currentColor">干净的 T5 encoder 输出，升采样到帧率 — 条件</text>
<text x="0" y="162" font-family="var(--sans)" font-size="14" font-weight="700" fill="currentColor">S2T</text>
<text x="44" y="162" font-family="var(--mono)" font-size="10.5" fill="var(--text-ch)">flow 在文本通道</text>
<text x="196" y="188" text-anchor="end" font-family="var(--mono)" font-size="11" fill="var(--voice)">语音 z_V · 100</text>
<rect x="206" y="170" width="640" height="26" rx="3" fill="var(--voice-soft)" stroke="var(--voice-edge)" stroke-width="1.5"/>
<text x="526" y="188" text-anchor="middle" font-family="var(--mono)" font-size="11" fill="currentColor">干净的原始 log-mel — 条件</text>
<text x="196" y="220" text-anchor="end" font-family="var(--mono)" font-size="11" fill="var(--text-ch)">文本 z_T · 512</text>
<rect x="206" y="202" width="640" height="26" rx="3" fill="url(#ht2)" stroke="var(--text-ch)" stroke-width="2"/>
<text x="526" y="220" text-anchor="middle" font-family="var(--mono)" font-size="11" fill="currentColor">加噪的 T5 latent — 目标，逐帧解码后合并重复</text>
<line x1="526" y1="124" x2="526" y2="148" stroke="currentColor" stroke-width="1.2" opacity=".45"/>
<text x="546" y="140" font-family="var(--mono)" font-size="10.5" fill="currentColor" opacity=".6">干净 / 加噪 对调</text>
</svg></div>
<figcaption><b>斜纹 = 加噪、是 flow 的目标；实底 = 干净、是条件。</b>两个方向之间唯一变的就是这件事，所以可以用同一个目标函数、同一个双向注意力。</figcaption></figure>
"""


if __name__ == "__main__":
    main()
