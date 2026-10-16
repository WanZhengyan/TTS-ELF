#!/usr/bin/env python
"""Where does the unified model lose it: the prediction, or the trajectory?

Run B at step 4000 (no guidance term, so l2/trivial is exactly the squared
relative x_hat0 error) reports

    teacher-forced x_hat0 rel err   TTS 0.483   S2T 0.667     both beat the mean
    rollout final err / ||x0||      TTS 1.078   S2T 1.521     both useless

So the per-step prediction is learning and the integrated trajectory is not.
Three things produce that and they need different fixes:

  1 exposure gap   x_hat0 is good on the true noisy path and bad on the path
                   the model itself generates. Fix: SDE sampling, more NFE,
                   self-conditioning at inference matching training.
  2 bad integration  x_hat0 is good everywhere but the ODE step is wrong
                   (schedule, t_eps clamping, noise_scale mismatch).
  3 low-t collapse  x_hat0 is only good at high t, where z is already nearly
                   clean, and useless at the low t the rollout starts from.

This measures x_hat0 error on BOTH paths across t, so the three are separable:
  1 shows roll >> tf at mid/high t
  2 shows roll ~ tf everywhere, yet the final z is far off
  3 shows tf itself rising to ~1.0 as t falls
"""
import argparse, os, sys

import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from unified import TTS, S2T, Cfg, UnifiedELF, net_to_v, restore
from train_unified import (LibriCache, TextEncoder, make_collate, _build_targets,
                           make_schedule)


def log(*a):
    print(*a, flush=True)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "uni_if25", "ckpt.pt"))
    ap.add_argument("--cache", default=os.path.join(HERE, "cache_libri_train_clean_100"))
    ap.add_argument("--weights", default="ema")
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--batch", type=int, default=12)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ta = ck["args"]
    log(f"ckpt {a.ckpt}")
    log(f"  step {ck['step']}  weights {a.weights}  init_frac "
        f"{ta.get('init_frac')}  p_s2t {ta['p_s2t']}  NFE {a.nfe}")

    ds = LibriCache(a.cache, "val")
    meta = ds.meta
    model = UnifiedELF(ta["model"], char_vocab=meta["char_vocab_size"],
                       max_length=ta["max_length"]).to(dev)
    model.load_state_dict(ck[a.weights]); model.eval()
    text_enc = TextEncoder(ta["text_latent_std"]).to(dev)
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("t5-small")
    itos = {v: k for k, v in meta["stoi"].items()}
    mel_stats = (meta["latent_mean"], meta["latent_std"])
    cfg = Cfg()

    for task, tname in ((TTS, "TTS"), (S2T, "S2T")):
        coll = make_collate(ta["max_length"], 1.0 if task == S2T else 0.0,
                            (0.3, 0.3) if task == TTS else (0.0, 0.0),
                            rng_seed=99, max_text=ta["max_text"],
                            itos=itos, tk=tk)
        b = next(iter(DataLoader(ds, batch_size=a.batch, collate_fn=coll)))
        is_text = b["is_text"].to(dev); valid = b["valid"].to(dev)
        prompt = b["prompt"].to(dev); chars = b["chars"].to(dev)
        tid = b["task"].to(dev)
        x0_t, x0_v = _build_targets(b, text_enc, dev, mel_stats)
        B = x0_t.shape[0]
        v_act = ((1.0 - is_text) * valid).unsqueeze(-1)
        t_act = (is_text * valid).unsqueeze(-1)
        tgt = (is_text if task == S2T else (1 - is_text))
        m = (tgt * valid * (1 - prompt)).unsqueeze(-1)
        x0 = x0_t if task == S2T else x0_v
        nrm = (x0 * m).norm().item()
        sc_cfg = torch.ones(B, device=dev)

        z_t = restore(torch.randn_like(x0_t) * cfg.denoiser_noise_scale, x0_t,
                      prompt) * t_act
        z_v = restore(torch.randn_like(x0_v) * cfg.denoiser_noise_scale, x0_v,
                      prompt) * v_act
        xp_t, xp_v = torch.zeros_like(z_t), torch.zeros_like(z_v)
        ts = make_schedule(a.nfe, cfg, dev)

        log("")
        log(f"=== {tname} ===   ||x0|| {nrm:.1f};  all errors are / ||x0||, so")
        log("1.00 = predicting the mean. z_err is how far the trajectory itself is.")
        log(f"{'t':>6}{'xhat0 roll':>12}{'xhat0 tf':>11}{'gap':>7}{'z_err':>9}")
        for i in range(a.nfe):
            tv = float(ts[i])
            t = torch.full((B,), tv, device=dev)
            xh_t, xh_v, _ = model(z_t, z_v, xp_t, xp_v, chars, t, tid,
                                  attention_mask=valid,
                                  self_cond_cfg_scale=sc_cfg)
            if i % max(a.nfe // 8, 1) == 0:
                tt = t.view(-1, 1, 1)
                tf_t = restore(tt * x0_t + (1 - tt) * torch.randn_like(x0_t)
                               * cfg.denoiser_noise_scale, x0_t, prompt) * t_act
                tf_v = restore(tt * x0_v + (1 - tt) * torch.randn_like(x0_v)
                               * cfg.denoiser_noise_scale, x0_v, prompt) * v_act
                yh_t, yh_v, _ = model(tf_t, tf_v, torch.zeros_like(tf_t),
                                      torch.zeros_like(tf_v), chars, t, tid,
                                      attention_mask=valid,
                                      self_cond_cfg_scale=sc_cfg)
                xh = xh_t if task == S2T else xh_v
                yh = yh_t if task == S2T else yh_v
                z = z_t if task == S2T else z_v
                er = ((xh - x0) * m).norm().item() / nrm
                et = ((yh - x0) * m).norm().item() / nrm
                ez = ((z - x0) * m).norm().item() / nrm
                log(f"{tv:>6.2f}{er:>12.3f}{et:>11.3f}"
                    f"{er/max(et,1e-9):>7.2f}{ez:>9.3f}")
            h = float(ts[i + 1]) - tv
            z_t = restore(z_t + h * net_to_v(xh_t, z_t, t), x0_t, prompt) * t_act
            z_v = restore(z_v + h * net_to_v(xh_v, z_v, t), x0_v, prompt) * v_act
            xp_t = restore(xh_t, x0_t, prompt) * t_act
            xp_v = restore(xh_v, x0_v, prompt) * v_act
        z = z_t if task == S2T else z_v
        log(f"  final z_err {((z - x0) * m).norm().item()/nrm:.3f}")

    log("")
    log("roll >> tf at mid t          -> exposure gap")
    log("roll ~ tf but final z_err ~1 -> the integration, not the prediction")
    log("tf itself ~1.0 at low t      -> nothing to integrate from")


if __name__ == "__main__":
    main()
