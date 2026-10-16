#!/usr/bin/env python
"""How much of the TTS training target is the model's own prediction?

train_step builds the target velocity as

    v_tgt = net_to_v(x0, z, t) + g * (v(model, sc on) - v(model, sc off))

with g = 1 - 1/sc_cfg, sc_cfg ~ U(1, 3), so g in [0, 0.67]. That second term is
ELF's self-conditioning-CFG sharpening: the model's own two predictions,
differenced. On ELF's text latents the predictions are good and the term
sharpens. On TTS the rollout error is 0.92 of ||x0|| -- near the mean -- so the
same term injects the model's own errors into its target. If it is a large
fraction of the true velocity, TTS is chasing a self-referential target.

Measured on the TTS rows and the S2T rows separately, on both the corpus where
TTS works (LJSpeech) and where it fails (LibriSpeech), at several t.
"""
import os
import sys

import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_unified as T
from unified import Cfg, UnifiedELF, S2T, TTS, net_to_v, restore


@torch.no_grad()
def measure(run, cache, max_text, dev, n_batch=2, bs=8):
    ck = torch.load(os.path.join(HERE, "runs", run, "ckpt.pt"),
                    map_location="cpu", weights_only=False)
    ta = ck["args"]
    ds = T.LibriCache(cache, "test" if "lj" in cache else "all")
    meta = ds.meta
    itos = {v: k for k, v in meta["stoi"].items()}
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("t5-small")
    wdim = meta.get("whisper_dim", 0) if ta.get("whisper") else 0
    wrate = meta.get("whisper_rate", 0.0) if ta.get("whisper") else 0.0
    model = UnifiedELF(ta["model"], char_vocab=meta["char_vocab_size"],
                       max_length=ta["max_length"], whisper_dim=wdim,
                       use_cond=not ta.get("no_cond", 0),
                       spk_dim=(meta.get("spk_emb_dim", 0)
                                if ta.get("spk") else 0)).to(dev)
    model.load_state_dict(ck["ema"]); model.eval()
    te = T.TextEncoder(ta["text_latent_std"]).to(dev)
    coll = T.make_collate(ta["max_length"], 0.5, (0.0, 0.3), rng_seed=11,
                          max_text=ta["max_text"], itos=itos, tk=tk,
                          layout=ta["layout"], whisper_rate=wrate,
                          frame_rate=meta["frame_rate"])
    cfg = Cfg(denoiser_p_mean=ta["denoiser_p_mean"])
    ms = (meta["latent_mean"], meta["latent_std"])
    ns = cfg.denoiser_noise_scale
    out = {}
    dl = DataLoader(ds, batch_size=bs, collate_fn=coll)
    for tv in (0.1, 0.25, 0.5, 0.75):
        acc = {"tts": [0.0, 0.0], "s2t": [0.0, 0.0]}
        for bi, b in enumerate(dl):
            if bi >= n_batch:
                break
            chars = b["chars"].to(dev); valid = b["valid"].to(dev)
            v_act = b["v_act"].to(dev).unsqueeze(-1)
            t_act = b["t_act"].to(dev).unsqueeze(-1)
            p_v, p_t = b["prompt_v"].to(dev), b["prompt_t"].to(dev)
            task = b["task"].to(dev)
            x0_t, x0_v = T.build_targets(b, te, dev, ms)
            ac, am = T.audio_cond(b, dev)
            se = T.spk_cond(b, dev)
            B = x0_t.shape[0]
            t = torch.full((B,), tv, device=dev)
            tt = t.view(-1, 1, 1)
            torch.manual_seed(bi)
            z_v = restore(tt * x0_v + (1 - tt) * torch.randn_like(x0_v) * ns,
                          x0_v, p_v) * v_act
            z_t = restore(tt * x0_t + (1 - tt) * torch.randn_like(x0_t) * ns,
                          x0_t, p_t) * t_act
            # the largest guidance weight used in training: g = 1 - 1/3
            sc = torch.full((B,), 3.0, device=dev)
            zt0, zv0 = torch.zeros_like(z_t), torch.zeros_like(z_v)
            ou_t, ou_v, _ = model(z_t, z_v, zt0, zv0, chars, t, task,
                                  attention_mask=valid, self_cond_cfg_scale=sc,
                                  audio_cond=ac, audio_cond_mask=am, spk_emb=se)
            xu_t = restore(ou_t, x0_t, p_t) * t_act
            xu_v = restore(ou_v, x0_v, p_v) * v_act
            oc_t, oc_v, _ = model(z_t, z_v, xu_t, xu_v, chars, t, task,
                                  attention_mask=valid, self_cond_cfg_scale=sc,
                                  audio_cond=ac, audio_cond_mask=am, spk_emb=se)
            g = 1.0 - 1.0 / 3.0
            for nm, x0, z, oc, ou, msk_task, act, prm in (
                    ("tts", x0_v, z_v, oc_v, ou_v, task == TTS, v_act, p_v),
                    ("s2t", x0_t, z_t, oc_t, ou_t, task == S2T, t_act, p_t)):
                tgt_m = (act.squeeze(-1) * (1 - prm))[msk_task].unsqueeze(-1)
                vtrue = net_to_v(x0, z, t)[msk_task] * tgt_m
                gui = (g * (net_to_v(oc, z, t) - net_to_v(ou, z, t)))[msk_task] * tgt_m
                acc[nm][0] += float(gui.pow(2).sum())
                acc[nm][1] += float(vtrue.pow(2).sum())
        out[tv] = {k: (v[0] / max(v[1], 1e-12)) ** 0.5 for k, v in acc.items()}
    del model, te
    torch.cuda.empty_cache()
    return out


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("  ||g·(v_sc_on - v_sc_off)|| / ||v_true||  on the TARGET positions, g=0.67")
    print("  (0 = target is the pure flow velocity; 1 = the model's own term is as")
    print("   large as the true velocity)")
    print()
    rows = [("LJSpeech  d_cond_fixed32", "d_cond_fixed32", "cache_lj_uni", 48),
            ("LibriSpeech libri_tc100_uni_ddp", "libri_tc100_uni_ddp",
             "cache_libri_test_clean", 112)]
    for name, run, cache, mt in rows:
        r = measure(run, os.path.join(HERE, cache), mt, dev)
        print(f"  {name}")
        print(f"    {'t':>6}{'TTS':>10}{'S2T':>10}")
        for tv, v in r.items():
            print(f"    {tv:6.2f}{v['tts']:10.3f}{v['s2t']:10.3f}")
        print()


if __name__ == "__main__":
    main()
