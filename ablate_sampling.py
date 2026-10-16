#!/usr/bin/env python
"""Sampling-side ablations: self-conditioning, step count K, SDE gamma.

All three are inference-only, so they run on the finished checkpoint with no
retraining. S2T is scored by WER and token accuracy, which need no vocoder and
are the decisive numbers for that direction.

Why these three. ELF-S2T's paper states its inference recipe as an SDE sampler
with K = 128 steps and audio classifier-free guidance w = 2.0, and its own
ablation puts K in {32, 64, 128} at 10.5-10.7% WER -- a 0.2 point spread, so
step count barely matters for them. We sample with an ODE at K = 32 and no
ACFG. This measures the two of those three we can test without retraining:

  self-conditioning  trained with self_cond_prob 0.5, so the model saw both
                     halves. Inference has always fed x_hat0 from the previous
                     step. Feeding zero instead says whether it earns its cost.
  K                  16 / 32 / 64 / 128, to see whether our 32 is leaving
                     anything on the table the way the paper's spread suggests.
  SDE gamma          0 (pure ODE) vs 1.5. Measured null once before, on a model
                     that was predicting the mean; worth redoing on one that is
                     not.

ACFG cannot be tested here. It needs an unconditional branch, which comes from
dropping the AUDIO condition during training; our label_drop_prob 0.1 drops the
upsampled-character block instead, so no such branch was learned.
"""
import argparse, json, os, sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from unified import TTS, S2T, Cfg, UnifiedELF, net_to_v, restore
from train_unified import (LibriCache, TextEncoder, make_collate, build_targets,
                           make_schedule, audio_cond, wer_pair, T5_EOS, T5_PAD)


def log(*a):
    print(*a, flush=True)


@torch.no_grad()
def s2t_run(model, b, cfg, dev, text_enc, mel_stats, tk, *, steps, use_sc,
            gamma=0.0, acfg_w=1.0):
    """acfg_w > 1 runs ELF-S2T's audio classifier-free guidance: two forwards
    per step, one with the audio condition and one with it removed, combined as
        v = v_uncond + w (v_cond - v_uncond)
    This only means anything if the model was trained with audio_drop_prob > 0,
    which is what creates the unconditional branch."""
    chars = b["chars"].to(dev); valid = b["valid"].to(dev)
    v_act = b["v_act"].to(dev).unsqueeze(-1)
    t_act = b["t_act"].to(dev).unsqueeze(-1)
    p_v, p_t = b["prompt_v"].to(dev), b["prompt_t"].to(dev)
    task = b["task"].to(dev)
    x0_t, x0_v = build_targets(b, text_enc, dev, mel_stats)
    ac, am = audio_cond(b, dev)
    B = x0_t.shape[0]
    ns = cfg.denoiser_noise_scale
    z_t = restore(torch.randn_like(x0_t) * ns, x0_t, p_t) * t_act
    z_v = restore(torch.randn_like(x0_v) * ns, x0_v, p_v) * v_act
    xp_t, xp_v = torch.zeros_like(z_t), torch.zeros_like(z_v)
    sc_cfg = torch.ones(B, device=dev)
    ts = make_schedule(steps, cfg, dev)
    for i in range(steps):
        tv = float(ts[i]); tn = float(ts[i + 1]); h = tn - tv
        if gamma > 0.0:
            al = max(0.0, 1.0 - gamma * h)
            z_t = restore(al * z_t + (1 - al) * torch.randn_like(z_t) * ns,
                          x0_t, p_t) * t_act
            z_v = restore(al * z_v + (1 - al) * torch.randn_like(z_v) * ns,
                          x0_v, p_v) * v_act
            tv = al * tv
        t = torch.full((B,), tv, device=dev)
        xh_t, xh_v, _ = model(z_t, z_v, xp_t, xp_v, chars, t, task,
                              attention_mask=valid, self_cond_cfg_scale=sc_cfg,
                              audio_cond=ac, audio_cond_mask=am)
        hh = tn - tv
        v_t = net_to_v(xh_t, z_t, t)
        if acfg_w != 1.0:
            # the unconditional branch: no Whisper features and no clean mel
            zv_u = torch.zeros_like(z_v)
            xu_t, _, _ = model(z_t, zv_u, xp_t, torch.zeros_like(xp_v), chars,
                               t, task, attention_mask=valid,
                               self_cond_cfg_scale=sc_cfg)
            v_u = net_to_v(xu_t, z_t, t)
            v_t = v_u + acfg_w * (v_t - v_u)
        z_t = restore(z_t + hh * v_t, x0_t, p_t) * t_act
        z_v = restore(z_v + hh * net_to_v(xh_v, z_v, t), x0_v, p_v) * v_act
        if use_sc:
            xp_t = restore(xh_t, x0_t, p_t) * t_act
            xp_v = restore(xh_v, x0_v, p_v) * v_act
        # else: stays zero, which is the half of training that dropped it

    t1 = torch.ones(B, device=dev)
    _, _, logits = model(z_t, z_v, xp_t, xp_v, chars, t1, task,
                         attention_mask=valid, self_cond_cfg_scale=sc_cfg,
                         decoder_step_active=True, audio_cond=ac,
                         audio_cond_mask=am)
    pred = logits.argmax(-1)
    err = ln = tok = tokn = 0
    for i in range(B):
        n = int(b["n_text"][i]); st = int(b["text_start"][i])
        ref_ids = [x for x in b["t5_dense"][i, :n].tolist()
                   if x not in (T5_EOS, T5_PAD)]
        raw = pred[i, st:st + b["t5_dense"].shape[1]].tolist()
        if T5_EOS in raw:
            raw = raw[:raw.index(T5_EOS)]
        hyp_ids = [x for x in raw if x != T5_PAD]
        e, l = wer_pair(tk.decode(ref_ids, skip_special_tokens=True),
                        tk.decode(hyp_ids, skip_special_tokens=True))
        err += e; ln += l
        k = min(len(ref_ids), len(hyp_ids))
        tok += sum(1 for j in range(k) if ref_ids[j] == hyp_ids[j])
        tokn += len(ref_ids)
    return 100.0 * err / max(ln, 1), tok / max(tokn, 1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "hy_wh_af", "ckpt.pt"))
    ap.add_argument("--cache", default=os.path.join(HERE, "cache_lj_uni"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n_batch", type=int, default=3)
    ap.add_argument("--acfg", type=int, default=0,
                    help="1 = also sweep the audio-guidance scale w")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "ablate_sampling.json"))
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
    log(f"ckpt step {ck['step']}  whisper={wdim}  decoder_p_mean="
        f"{ta.get('decoder_p_mean')}  n={a.batch * a.n_batch} clips")

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
    log("=== S2T · 自条件开关 × 步数 K × SDE gamma ===")
    log(f"{'use_sc':>7}{'K':>6}{'gamma':>7}{'WER %':>9}{'tok acc':>9}")
    for use_sc in (True, False):
        for K in (16, 32, 64, 128):
            for g in (0.0, 1.5) if K == 32 else (0.0,):
                we, ac = [], []
                for b in batches:
                    w, t = s2t_run(model, b, cfg, dev, text_enc, mel_stats, tk,
                                   steps=K, use_sc=use_sc, gamma=g)
                    we.append(w); ac.append(t)
                W, A = float(np.mean(we)), float(np.mean(ac))
                res[f"sc{int(use_sc)}_K{K}_g{g}"] = {"wer": W, "tok_acc": A}
                log(f"{str(use_sc):>7}{K:>6}{g:>7.1f}{W:>9.2f}{A*100:>8.2f}%")

    if a.acfg:
        log("")
        log("=== S2T · 音频引导 ACFG（每步两次 forward）===")
        log(f"{'w':>6}{'K':>6}{'WER %':>9}{'tok acc':>9}{'forward 次数':>13}")
        for w in (1.0, 1.5, 2.0, 2.5, 3.0):
            we, acc = [], []
            for b in batches:
                ww, tt = s2t_run(model, b, cfg, dev, text_enc, mel_stats, tk,
                                 steps=32, use_sc=True, acfg_w=w)
                we.append(ww); acc.append(tt)
            W, A = float(np.mean(we)), float(np.mean(acc))
            res[f"acfg_w{w}_K32"] = {"wer": W, "tok_acc": A}
            log(f"{w:>6.1f}{32:>6}{W:>9.2f}{A*100:>8.2f}%"
                f"{(32 if w == 1.0 else 64):>12}")
        log("  w = 1.0 就是无引导，和上表的 sc1_K32_g0.0 应当一致")

    base = res["sc1_K32_g0.0"]
    log("")
    log("参照：我们一直用的设定 = 自条件开 · K=32 · ODE")
    log(f"  WER {base['wer']:.2f}%  tok_acc {base['tok_acc']*100:.2f}%")
    for k, v in sorted(res.items(), key=lambda kv: kv[1]["wer"]):
        log(f"  {k:<16} WER {v['wer']:>7.2f}  ({v['wer']-base['wer']:+.2f})"
            f"   acc {v['tok_acc']*100:>6.2f}%")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
