#!/usr/bin/env python
"""Has the rollout trajectory collapsed, and does SDE sampling fix it?

Five configurations (token vs channel layout, mixed vs pure TTS, zero vs 0.25
init of the new input columns, denoiser_p_mean -1.5 vs 0.0) all land on a TTS
rollout error of 0.85-0.87 against a trivial baseline of 1.000, while the
teacher-forced x_hat0 error improves normally (0.86 -> 0.61 as t rises). The
rollout error is FLAT in t, which is the tell: the ODE

    z <- z + h (x_hat0 - z) / (1 - t)

transports z onto x_hat0, and x_hat0 from an L2 objective is the conditional
mean, whose norm is shrunk. A conditional mean sitting 0.86 away has norm
sqrt(1 - 0.86^2) = 0.51 of ||x0||, which reproduces an error of 0.86 -- self
consistent. Flow matching normally escapes this because E[x0 | z_t] sharpens as
t rises, but that only happens if z stays on the training marginal
t*x0 + (1-t)*eps*noise_scale. Pulled toward a shrunk mean, z gets NARROWER than
that, so the model sees out-of-distribution inputs and its estimate never
sharpens.

Two things are measured here, neither needing any retraining:

  1 ||z|| along the rollout against ||z|| of the training marginal at the same
    t. Collapse shows up directly as a norm deficit.
  2 ODE against SDE at several gamma. ELF-S2T samples with SDE gamma = 1.5;
    this file's rollout was ODE only, which is an omission, not a choice. The
    SDE step re-injects noise each step, which is exactly what keeps z on the
    training marginal:
        alpha = 1 - gamma*h;  z <- alpha*z + (1-alpha)*eps*noise_scale
        t_cur = alpha*t;      z <- z + (t_next - t_cur) * v
"""
import argparse, os, sys

import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from unified import TTS, S2T, Cfg, UnifiedELF, net_to_v, restore
from train_unified import (LibriCache, TextEncoder, make_collate, build_targets,
                           make_schedule, collapse, wer_pair, T5_EOS, T5_PAD)


def log(*a):
    print(*a, flush=True)


@torch.no_grad()
def run(model, batch, cfg, device, mel_stats, text_enc, task_sel,
        steps=32, gamma=0.0, acfg=1.0, trace=False):
    chars = batch["chars"].to(device); valid = batch["valid"].to(device)
    v_act = batch["v_act"].to(device).unsqueeze(-1)
    t_act = batch["t_act"].to(device).unsqueeze(-1)
    p_v, p_t = batch["prompt_v"].to(device), batch["prompt_t"].to(device)
    task = batch["task"].to(device)
    x0_t, x0_v = build_targets(batch, text_enc, device, mel_stats)
    B = x0_t.shape[0]
    ns = cfg.denoiser_noise_scale
    s2t = task_sel == S2T
    x0 = x0_t if s2t else x0_v
    m = ((batch["t_act"] if s2t else batch["v_act"]).to(device)
         * (1 - (p_t if s2t else p_v))).unsqueeze(-1)
    nrm = (x0 * m).norm().item()

    z_t = restore(torch.randn_like(x0_t) * ns, x0_t, p_t) * t_act
    z_v = restore(torch.randn_like(x0_v) * ns, x0_v, p_v) * v_act
    xp_t, xp_v = torch.zeros_like(z_t), torch.zeros_like(z_v)
    sc_cfg = torch.full((B,), acfg, device=device)
    ts = make_schedule(steps, cfg, device)
    tr = []
    for i in range(steps):
        tv = float(ts[i]); tn = float(ts[i + 1]); h = tn - tv
        if gamma > 0.0:
            # SDE: shrink toward noise, then take a longer deterministic step
            al = max(0.0, 1.0 - gamma * h)
            z_t = restore(al * z_t + (1 - al) * torch.randn_like(z_t) * ns,
                          x0_t, p_t) * t_act
            z_v = restore(al * z_v + (1 - al) * torch.randn_like(z_v) * ns,
                          x0_v, p_v) * v_act
            tv_eff = al * tv
        else:
            tv_eff = tv
        t = torch.full((B,), tv_eff, device=device)
        xh_t, xh_v, _ = model(z_t, z_v, xp_t, xp_v, chars, t, task,
                              attention_mask=valid, self_cond_cfg_scale=sc_cfg)
        hh = tn - tv_eff
        z_t = restore(z_t + hh * net_to_v(xh_t, z_t, t), x0_t, p_t) * t_act
        z_v = restore(z_v + hh * net_to_v(xh_v, z_v, t), x0_v, p_v) * v_act
        xp_t = restore(xh_t, x0_t, p_t) * t_act
        xp_v = restore(xh_v, x0_v, p_v) * v_act
        if trace and i % max(steps // 8, 1) == 0:
            z = z_t if s2t else z_v
            xh = xh_t if s2t else xh_v
            # the training marginal's norm at this t, for the same mask
            want = ((tv * x0 + (1 - tv) * torch.randn_like(x0) * ns) * m).norm().item()
            tr.append((tv, (z * m).norm().item() / max(want, 1e-9),
                       (xh * m).norm().item() / max(nrm, 1e-9),
                       ((z - x0) * m).norm().item() / max(nrm, 1e-9)))
    z = z_t if s2t else z_v
    t1 = torch.ones(B, device=device)
    _, _, logits = model(z_t, z_v, xp_t, xp_v, chars, t1, task,
                         attention_mask=valid, self_cond_cfg_scale=sc_cfg,
                         decoder_step_active=True)
    return ((z - x0) * m).norm().item() / max(nrm, 1e-9), tr, logits


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
    log(f"ckpt {a.ckpt}")
    log(f"  step {ck['step']}  weights {a.weights}  layout {ta['layout']}  "
        f"p_s2t {ta['p_s2t']}  init_frac {ta.get('init_frac')}  NFE {a.nfe}")

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

    tasks = [(TTS, "TTS", 0.0, (0.3, 0.3))]
    if ta["p_s2t"] > 0:
        tasks.append((S2T, "S2T", 1.0, (0.0, 0.0)))

    for task_sel, tname, ps2t, pf in tasks:
        coll = make_collate(ta["max_length"], ps2t, pf, rng_seed=99,
                            max_text=ta["max_text"], itos=itos, tk=tk,
                            layout=ta["layout"])
        b = next(iter(DataLoader(ds, batch_size=a.batch, collate_fn=coll)))

        log("")
        log(f"================ {tname} ================")
        log("1. has the trajectory collapsed?  ODE, gamma = 0")
        log(f"{'t':>6}{'|z| / training marginal':>26}{'|xhat0|/|x0|':>14}"
            f"{'z_err':>9}")
        _, tr, _ = run(model, b, cfg, dev, mel_stats, text_enc, task_sel,
                       steps=a.nfe, gamma=0.0, trace=True)
        for tv, zr, xr, ze in tr:
            log(f"{tv:>6.2f}{zr:>26.3f}{xr:>14.3f}{ze:>9.3f}")
        log("  a norm ratio well under 1.0 means z is narrower than anything the")
        log("  model was trained on, so its own trajectory is off-distribution.")

        log("")
        log("2. does SDE put it back?  final z_err / ||x0||, trivial = 1.000")
        log(f"{'gamma':>7}{'z_err':>9}   {'':>4}")
        best = None
        for g in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
            e, _, lg = run(model, b, cfg, dev, mel_stats, text_enc, task_sel,
                           steps=a.nfe, gamma=g)
            mark = "  <- ELF-S2T's setting" if abs(g - 1.5) < 1e-9 else ""
            log(f"{g:>7.1f}{e:>9.4f}{mark}")
            if best is None or e < best[1]:
                best = (g, e)
            if task_sel == S2T:
                pred = lg.argmax(-1)
                err = ln = 0
                for i in range(pred.shape[0]):
                    n = int(b["n_text"][i])
                    ref_ids = [x for x in b["t5_dense"][i, :n].tolist()
                               if x not in (T5_EOS, T5_PAD)]
                    hyp = collapse(pred[i].tolist()) if ta["layout"] == "channel" \
                        else [x for x in pred[i].tolist() if x != T5_PAD]
                    ee, ll = wer_pair(tk.decode(ref_ids, skip_special_tokens=True),
                                      tk.decode(hyp, skip_special_tokens=True))
                    err += ee; ln += ll
                log(f"         WER {100*err/max(ln,1):.2f}%")
        log(f"  best gamma {best[0]} at {best[1]:.4f}  "
            f"({'beats' if best[1] < 1 else 'does NOT beat'} generating nothing)")


if __name__ == "__main__":
    main()
