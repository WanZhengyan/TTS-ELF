#!/usr/bin/env python
"""Is the conditioning reaching the trunk at all?

At step 2000 the unified model is good teacher-forced (l2/trivial 0.24, decode
acc 92%) and useless in rollout (TTS mel err/trivial 0.89-3.74, S2T WER 101-263%,
token acc 0.7%). Two things could produce that, and they call for opposite fixes:

  A  the conditioning pathway is still near zero, so the model is generating
     from ELF's unconditional openwebtext prior and ignoring the audio/text
  B  the pathway is live but the flow cannot roll out (an exposure gap)

Both are measurable, so neither needs to be assumed.

COLUMN NORMS. self_cond_proj's input is [z_T | sc_T | z_V | sc_V | cond] =
512+512+100+100+512. ELF's pretrained weights filled the first 1024 columns;
load_elf_init ZEROED the remaining 712 on purpose, so that the model starts at
ELF's behaviour. If those columns are still tiny next to the pretrained ones,
the audio and the upsampled text are barely driving the trunk.

CONDITION SENSITIVITY. Zero the condition and see how much the prediction
moves. A model that ignores its condition moves ~0%.
"""
import os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from unified import TTS, S2T, Cfg, UnifiedELF, net_to_v, restore
from train_unified import LibriCache, TextEncoder, make_collate, _build_targets


def log(*a):
    print(*a, flush=True)


@torch.no_grad()
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "uni_b32", "ckpt.pt"))
    ap.add_argument("--cache", default=os.path.join(HERE, "cache_libri_train_clean_100"))
    ap.add_argument("--weights", default="ema")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ta = ck["args"]
    log(f"ckpt step {ck['step']}  weights={a.weights}  "
        f"max_length {ta['max_length']}  max_text {ta['max_text']}")

    ds = LibriCache(a.cache, "val")
    meta = ds.meta
    model = UnifiedELF(ta["model"], char_vocab=meta["char_vocab_size"],
                       max_length=ta["max_length"]).to(dev)
    model.load_state_dict(ck[a.weights]); model.eval()

    # ---------- 1. are the new input columns live?
    W = model.elf.self_cond_proj.weight.data.float()      # (512, 1736)
    blocks = [("z_T   (ELF pretrained)", 0, 512),
              ("sc_T  (ELF pretrained)", 512, 1024),
              ("z_V   (new, zeroed)   ", 1024, 1124),
              ("sc_V  (new, zeroed)   ", 1124, 1224),
              ("cond  (new, zeroed)   ", 1224, 1736)]
    log("")
    log("self_cond_proj column norms, per input column (so widths are comparable)")
    ref = None
    for name, lo, hi in blocks:
        w = W[:, lo:hi]
        per_col = w.norm(dim=0).mean().item()
        if ref is None:
            ref = per_col
        log(f"  {name}  cols {hi-lo:>4}  |w|/col {per_col:.5f}  "
            f"= {per_col/ref*100:6.2f}% of z_T")

    # ---------- 2. does zeroing the condition change the prediction?
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("t5-small")
    itos = {v: k for k, v in meta["stoi"].items()}
    cfg = Cfg()
    text_enc = TextEncoder(ta["text_latent_std"]).to(dev)
    mel_stats = (meta["latent_mean"], meta["latent_std"])

    log("")
    log("condition sensitivity: how far the x_hat0 prediction moves when the")
    log("condition is zeroed. 0% means the condition is being ignored.")
    log(f"{'task':<6}{'t':>7}{'rel move':>11}{'|xhat| cond':>13}{'|xhat| none':>13}")
    for task, tname in ((S2T, "S2T"), (TTS, "TTS")):
        coll = make_collate(ta["max_length"], 1.0 if task == S2T else 0.0,
                            (0.3, 0.3) if task == TTS else (0.0, 0.0),
                            rng_seed=7, max_text=ta["max_text"],
                            itos=itos, tk=tk)
        from torch.utils.data import DataLoader
        b = next(iter(DataLoader(ds, batch_size=8, collate_fn=coll)))
        is_text = b["is_text"].to(dev); valid = b["valid"].to(dev)
        prompt = b["prompt"].to(dev); chars = b["chars"].to(dev)
        tid = b["task"].to(dev)
        x0_t, x0_v = _build_targets(b, text_enc, dev, mel_stats)
        B = x0_t.shape[0]
        v_act = ((1 - is_text) * valid).unsqueeze(-1)
        t_act = (is_text * valid).unsqueeze(-1)
        sc_cfg = torch.ones(B, device=dev)
        for t_val in (0.1, 0.4, 0.7):
            t = torch.full((B,), t_val, device=dev)
            tt = t.view(-1, 1, 1)
            n_t = torch.randn_like(x0_t) * cfg.denoiser_noise_scale
            n_v = torch.randn_like(x0_v) * cfg.denoiser_noise_scale
            z_t = restore(tt * x0_t + (1 - tt) * n_t, x0_t, prompt) * t_act
            z_v = restore(tt * x0_v + (1 - tt) * n_v, x0_v, prompt) * v_act
            zt_, zv_ = torch.zeros_like(z_t), torch.zeros_like(z_v)
            # with the condition
            a1_t, a1_v, _ = model(z_t, z_v, zt_, zv_, chars, t, tid,
                                  attention_mask=valid, self_cond_cfg_scale=sc_cfg)
            # condition removed: S2T loses the audio, TTS loses both text routes
            if task == S2T:
                a2_t, a2_v, _ = model(z_t, torch.zeros_like(z_v), zt_, zv_, chars,
                                      t, tid, attention_mask=valid,
                                      self_cond_cfg_scale=sc_cfg)
                out1, out2 = a1_t, a2_t
                m = (is_text * valid * (1 - prompt)).unsqueeze(-1)
            else:
                a2_t, a2_v, _ = model(torch.zeros_like(z_t), z_v, zt_, zv_,
                                      torch.zeros_like(chars), t, tid,
                                      attention_mask=valid,
                                      self_cond_cfg_scale=sc_cfg,
                                      drop_cond=torch.ones(B, device=dev))
                out1, out2 = a1_v, a2_v
                m = ((1 - is_text) * valid * (1 - prompt)).unsqueeze(-1)
            move = ((out1 - out2) * m).norm().item() / max(
                (out1 * m).norm().item(), 1e-9)
            log(f"{tname:<6}{t_val:>7.2f}{move*100:>10.2f}%"
                f"{(out1*m).norm().item():>13.2f}{(out2*m).norm().item():>13.2f}")

    log("")
    log("If the new column norms are a few percent of z_T and the sensitivity is")
    log("near zero, the diagnosis is A: the conditioning is not driving the trunk")
    log("and the zero init is the bottleneck, not the amount of training.")


if __name__ == "__main__":
    main()
