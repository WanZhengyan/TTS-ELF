#!/usr/bin/env python
"""Is x_hat0 simply half the amplitude it should be, and does a gain fix it?

Measured on the pure-TTS channel run at step 2000, ODE rollout:

    t      |z| / training marginal   |xhat0| / ||x0||
    0.00                     0.973              0.490
    0.50                     0.885              0.483
    0.88                     0.501              0.441

The trajectory collapses, but the reason is the second column: x_hat0 carries
~48% of the right norm at EVERY t, and SDE sampling (gamma 0 to 3) moved the
final error only from 0.8140 to 0.8185, so re-injecting noise is not the fix.

Conditional-mean shrinkage would shrink at low t and vanish as t -> 1, because
z_t -> x0. Flat 0.44-0.49 is not that. Two measurements separate the remaining
possibilities, neither needing retraining:

  1 the same norm ratio TEACHER-FORCED. If it is also ~0.5, the output scale is
    wrong everywhere and the objective or the output path is at fault. If it is
    ~1.0, the shrinkage belongs to the rollout alone.

  2 a gain on x_hat0 during the rollout. If x_hat0 ~= c*x0 with c ~= 0.5 and the
    direction is right, then dividing by c should fix the rollout outright, and
    the best gain names c. If no gain helps, the direction is wrong too and
    amplitude was never the problem.
"""
import argparse, os, sys

import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from unified import TTS, S2T, Cfg, UnifiedELF, net_to_v, restore
from train_unified import (LibriCache, TextEncoder, make_collate, build_targets,
                           make_schedule)


def log(*a):
    print(*a, flush=True)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "ch_tts", "ckpt.pt"))
    ap.add_argument("--cache", default=os.path.join(HERE, "cache_libri_train_clean_100"))
    ap.add_argument("--weights", default="ema")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--nfe", type=int, default=32)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ta = ck["args"]
    log(f"ckpt step {ck['step']}  weights {a.weights}  layout {ta['layout']}  "
        f"p_s2t {ta['p_s2t']}  NFE {a.nfe}")

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
    cfg = Cfg(denoiser_p_mean=ta["denoiser_p_mean"])

    coll = make_collate(ta["max_length"], 0.0, (0.3, 0.3), rng_seed=99,
                        max_text=ta["max_text"], itos=itos, tk=tk,
                        layout=ta["layout"])
    b = next(iter(DataLoader(ds, batch_size=a.batch, collate_fn=coll)))
    chars = b["chars"].to(dev); valid = b["valid"].to(dev)
    v_act = b["v_act"].to(dev).unsqueeze(-1)
    t_act = b["t_act"].to(dev).unsqueeze(-1)
    p_v, p_t = b["prompt_v"].to(dev), b["prompt_t"].to(dev)
    task = b["task"].to(dev)
    x0_t, x0_v = build_targets(b, text_enc, dev, mel_stats)
    B = x0_v.shape[0]
    ns = cfg.denoiser_noise_scale
    m = (b["v_act"].to(dev) * (1 - p_v)).unsqueeze(-1)
    nrm = (x0_v * m).norm().item()
    sc_cfg = torch.ones(B, device=dev)

    # ---- 1. teacher-forced norm ratio and the alignment behind it
    log("")
    log("1. TEACHER-FORCED, on the true noisy path")
    log(f"{'t':>6}{'|xhat0|/|x0|':>14}{'cos(xhat0,x0)':>16}"
        f"{'err':>8}{'err if rescaled':>18}")
    for tv in (0.0, 0.25, 0.5, 0.75, 0.9):
        t = torch.full((B,), tv, device=dev)
        tt = t.view(-1, 1, 1)
        zt = restore(tt * x0_t + (1 - tt) * torch.randn_like(x0_t) * ns,
                     x0_t, p_t) * t_act
        zv = restore(tt * x0_v + (1 - tt) * torch.randn_like(x0_v) * ns,
                     x0_v, p_v) * v_act
        _, xh, _ = model(zt, zv, torch.zeros_like(zt), torch.zeros_like(zv),
                         chars, t, task, attention_mask=valid,
                         self_cond_cfg_scale=sc_cfg)
        A, Bv = xh * m, x0_v * m
        na, nb = A.norm().item(), Bv.norm().item()
        cos = (A * Bv).sum().item() / max(na * nb, 1e-9)
        err = (A - Bv).norm().item() / max(nb, 1e-9)
        # if only the amplitude were wrong, this is what the error would become
        k = (A * Bv).sum().item() / max((A * A).sum().item(), 1e-9)
        err_rs = (k * A - Bv).norm().item() / max(nb, 1e-9)
        log(f"{tv:>6.2f}{na/max(nb,1e-9):>14.3f}{cos:>16.3f}"
            f"{err:>8.3f}{err_rs:>18.3f}")
    log("  cos is the part a gain cannot fix: err >= sqrt(1 - cos^2) whatever")
    log("  the amplitude. 'err if rescaled' is that floor, measured.")

    # ---- 2. a gain on xhat0 during the rollout
    log("")
    log("2. ROLLOUT with a gain on xhat0.  final z_err / ||x0||, trivial = 1.000")
    log(f"{'gain':>6}{'z_err':>9}{'|z| final / |x0|':>19}")
    best = None
    for gain in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0):
        z_t = restore(torch.randn_like(x0_t) * ns, x0_t, p_t) * t_act
        z_v = restore(torch.randn_like(x0_v) * ns, x0_v, p_v) * v_act
        xp_t, xp_v = torch.zeros_like(z_t), torch.zeros_like(z_v)
        ts = make_schedule(a.nfe, cfg, dev)
        for i in range(a.nfe):
            tv = float(ts[i])
            t = torch.full((B,), tv, device=dev)
            xh_t, xh_v, _ = model(z_t, z_v, xp_t, xp_v, chars, t, task,
                                  attention_mask=valid,
                                  self_cond_cfg_scale=sc_cfg)
            xh_v = xh_v * gain
            h = float(ts[i + 1]) - tv
            z_t = restore(z_t + h * net_to_v(xh_t, z_t, t), x0_t, p_t) * t_act
            z_v = restore(z_v + h * net_to_v(xh_v, z_v, t), x0_v, p_v) * v_act
            xp_t = restore(xh_t, x0_t, p_t) * t_act
            xp_v = restore(xh_v, x0_v, p_v) * v_act
        e = ((z_v - x0_v) * m).norm().item() / max(nrm, 1e-9)
        zn = (z_v * m).norm().item() / max(nrm, 1e-9)
        log(f"{gain:>6.2f}{e:>9.4f}{zn:>19.3f}")
        if best is None or e < best[1]:
            best = (gain, e)
    log(f"  best gain {best[0]} at {best[1]:.4f}")
    log("")
    log("gain fixes it            -> the amplitude was the whole problem")
    log("gain does nothing, cos low -> the direction is wrong; amplitude was a symptom")


if __name__ == "__main__":
    main()
