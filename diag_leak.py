#!/usr/bin/env python
"""Did S2T read the answer off the character block instead of the audio?

A leak found by inspection, not by a failing test. In make_collate the line

    chars[b, :T] = ch_full[a:a + T]

is NOT gated by task. t5_idx is gated, so z_T's audio region is correctly zero
for S2T rows -- but chars is not, so every S2T row receives the upsampled
CHARACTER SEQUENCE OF ITS OWN ANSWER through the cond block.

Whether that matters depends on whether the model learned to use it. It did
learn the pathway: self_cond_proj's cond columns sit at 51.7-55.5% of the
pretrained z_T block's per-column norm across every checkpoint, so the block is
far from inert.

The decisive test is inference-only: zero chars and re-measure S2T. If WER
collapses, the reported numbers were reading the answer. Training did drop the
cond block on 10% of rows (label_drop_prob), so the model has seen it absent --
which is why this is a measurement and not a foregone conclusion.
"""
import argparse, json, os, sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from unified import S2T, Cfg, UnifiedELF, net_to_v, restore
from train_unified import (LibriCache, TextEncoder, make_collate, build_targets,
                           make_schedule, audio_cond, wer_pair, T5_EOS, T5_PAD)


def log(*a):
    print(*a, flush=True)


@torch.no_grad()
def run(model, b, cfg, dev, text_enc, mel_stats, tk, *, steps, zero_chars,
        zero_audio=False):
    chars = b["chars"].to(dev)
    if zero_chars:
        chars = torch.zeros_like(chars)
    valid = b["valid"].to(dev)
    v_act = b["v_act"].to(dev).unsqueeze(-1)
    t_act = b["t_act"].to(dev).unsqueeze(-1)
    p_v, p_t = b["prompt_v"].to(dev), b["prompt_t"].to(dev)
    task = b["task"].to(dev)
    x0_t, x0_v = build_targets(b, text_enc, dev, mel_stats)
    ac, am = audio_cond(b, dev)
    if zero_audio:
        x0_v = torch.zeros_like(x0_v)
        if ac is not None:
            ac = torch.zeros_like(ac)
    B = x0_t.shape[0]
    ns = cfg.denoiser_noise_scale
    z_t = restore(torch.randn_like(x0_t) * ns, x0_t, p_t) * t_act
    z_v = restore(torch.randn_like(x0_v) * ns, x0_v, p_v) * v_act
    xp_t, xp_v = torch.zeros_like(z_t), torch.zeros_like(z_v)
    sc = torch.ones(B, device=dev)
    ts = make_schedule(steps, cfg, dev)
    for i in range(steps):
        t = torch.full((B,), float(ts[i]), device=dev)
        xh_t, xh_v, _ = model(z_t, z_v, xp_t, xp_v, chars, t, task,
                              attention_mask=valid, self_cond_cfg_scale=sc,
                              audio_cond=ac, audio_cond_mask=am)
        h = float(ts[i + 1]) - float(ts[i])
        z_t = restore(z_t + h * net_to_v(xh_t, z_t, t), x0_t, p_t) * t_act
        z_v = restore(z_v + h * net_to_v(xh_v, z_v, t), x0_v, p_v) * v_act
        xp_t = restore(xh_t, x0_t, p_t) * t_act
        xp_v = restore(xh_v, x0_v, p_v) * v_act
    t1 = torch.ones(B, device=dev)
    _, _, lg = model(z_t, z_v, xp_t, xp_v, chars, t1, task,
                     attention_mask=valid, self_cond_cfg_scale=sc,
                     decoder_step_active=True, audio_cond=ac,
                     audio_cond_mask=am)
    pred = lg.argmax(-1)
    err = ln = tok = tokn = 0
    samples = []
    for i in range(B):
        n = int(b["n_text"][i]); st = int(b["text_start"][i])
        ref_ids = [x for x in b["t5_dense"][i, :n].tolist()
                   if x not in (T5_EOS, T5_PAD)]
        raw = pred[i, st:st + b["t5_dense"].shape[1]].tolist()
        if T5_EOS in raw:
            raw = raw[:raw.index(T5_EOS)]
        hyp_ids = [x for x in raw if x != T5_PAD]
        r = tk.decode(ref_ids, skip_special_tokens=True)
        hp = tk.decode(hyp_ids, skip_special_tokens=True)
        e, l = wer_pair(r, hp)
        err += e; ln += l
        k = min(len(ref_ids), len(hyp_ids))
        tok += sum(1 for j in range(k) if ref_ids[j] == hyp_ids[j])
        tokn += len(ref_ids)
        if len(samples) < 2:
            samples.append((r, hp))
    return 100.0 * err / max(ln, 1), tok / max(tokn, 1), samples


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "p_dp5", "ckpt.pt"))
    ap.add_argument("--cache", default=os.path.join(HERE, "cache_lj_uni"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n_batch", type=int, default=6)
    ap.add_argument("--nfe", type=int, default=32)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ta = ck["args"]
    ds = LibriCache(a.cache, "test")
    meta = ds.meta
    wdim = meta.get("whisper_dim", 0) if ta.get("whisper") else 0
    wrate = meta.get("whisper_rate", 0.0) if ta.get("whisper") else 0.0
    model = UnifiedELF(ta["model"], char_vocab=meta["char_vocab_size"],
                       max_length=ta["max_length"], whisper_dim=wdim).to(dev)
    model.load_state_dict(ck["ema"]); model.eval()
    text_enc = TextEncoder(ta["text_latent_std"]).to(dev)
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("t5-small")
    itos = {v: k for k, v in meta["stoi"].items()}
    cfg = Cfg(denoiser_p_mean=ta["denoiser_p_mean"],
              decoder_p_mean=ta.get("decoder_p_mean", 0.8))
    mel_stats = (meta["latent_mean"], meta["latent_std"])
    log(f"ckpt {os.path.basename(os.path.dirname(a.ckpt))} step {ck['step']}  "
        f"whisper={wdim}  n={a.batch*a.n_batch} clips  NFE {a.nfe}")

    coll = make_collate(ta["max_length"], 1.0, (0.0, 0.0), rng_seed=11,
                        max_text=ta["max_text"], itos=itos, tk=tk,
                        layout=ta["layout"], whisper_rate=wrate,
                        frame_rate=meta["frame_rate"])
    batches = []
    for i, b in enumerate(DataLoader(ds, batch_size=a.batch, collate_fn=coll)):
        if i >= a.n_batch:
            break
        batches.append(b)

    res = {}
    log("")
    log(f"{'设定':<28}{'WER %':>9}{'tok acc':>10}")
    for name, kw in (("原样（chars 泄漏在）", dict(zero_chars=False)),
                     ("chars 置零", dict(zero_chars=True)),
                     ("音频置零、chars 保留", dict(zero_chars=False, zero_audio=True)),
                     ("两个都置零", dict(zero_chars=True, zero_audio=True))):
        we, acc, samp = [], [], None
        for b in batches:
            w, t, s = run(model, b, cfg, dev, text_enc, mel_stats, tk,
                          steps=a.nfe, **kw)
            we.append(w); acc.append(t)
            if samp is None:
                samp = s
        W, A = float(np.mean(we)), float(np.mean(acc))
        res[name] = {"wer": W, "tok_acc": A, "samples": samp}
        log(f"{name:<28}{W:>9.2f}{A*100:>9.2f}%")
        for r, h in samp:
            log(f"      ref  {r[:66]}")
            log(f"      hyp  {h[:66]}")

    base = res["原样（chars 泄漏在）"]["wer"]
    noc = res["chars 置零"]["wer"]
    noa = res["音频置零、chars 保留"]["wer"]
    log("")
    log(f"chars 置零后 WER {noc - base:+.2f}  ({base:.2f} -> {noc:.2f})")
    log(f"音频置零后 WER {noa - base:+.2f}  ({base:.2f} -> {noa:.2f})")
    log("")
    if noc - base > 15:
        log("-> chars 置零就崩：泄漏是承重的，报出来的 S2T 数字不可信")
    elif noc - base > 5:
        log("-> chars 置零有明显代价：泄漏被部分利用，数字被高估")
    else:
        log("-> chars 置零几乎无代价：模型主要在用音频，泄漏基本没被利用")
    if noa - base < 15:
        log("-> 但音频置零也不崩，那才是真的坏消息（说明它根本没在听）")
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "samples"}
               for k, v in res.items()},
              open(os.path.join(HERE, "out", "leak.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
