#!/usr/bin/env python
"""Separate "the flow did not converge" from "the discretization head fails".

Four independent measurements on held-out clips:

  A  teacher-forced denoiser quality — feed the TRUE z_t = t*x0 + (1-t)*eps*s and
     ask for x0_hat. Compares against two trivial baselines (predict 0, predict z_t).
     If the model cannot beat predict-z_t, it never learned the velocity field and
     nothing downstream can work.

  B  rollout quality — run the ODE sampler and record ||z_t - x0|| at every step.
     Distinguishes a learned-but-unintegrable field from an unlearned one.

  C  code accuracy on the SAME tensor, nearest-codebook vs the CE head, evaluated on
     (i) the rollout's final z, (ii) the model's own x0_hat. ELF decodes from z; if
     x0_hat decodes much better, that is a free fix.

  D  CE-head accuracy vs lambda over the distribution it was actually TRAINED on,
     lambda ~ sigmoid(N(0.8,0.8)). Answers whether the sigma-from-clean sweep was
     simply the wrong yardstick.
"""
import argparse, json, os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train import (ELFTTS, LJCache, make_collate, Cfg, load_codebook,  # noqa: E402
                   get_sampling_steps, restore, net_to_v, T_EPS)
from torch.utils.data import DataLoader  # noqa: E402


def log(*a):
    print(*a, flush=True)


def nn_codes(z, cb, cb_sq):
    """Nearest-codebook ids for z (…,D) with cb already in /latent_std units."""
    flat = z.reshape(-1, z.shape[-1])
    d = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ cb.t() + cb_sq[None, :]
    return d.argmin(1).view(z.shape[:-1])


def acc(pred, code, m):
    return (((pred == code).float() * m).sum() / m.sum().clamp(min=1)).item()


def rel(a, b, m):
    """Mean per-frame ||a-b|| over masked frames."""
    n = (a - b).norm(dim=-1)
    return ((n * m).sum() / m.sum().clamp(min=1)).item()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "test2", "ckpt.pt"))
    ap.add_argument("--cache", default=os.path.join(HERE, "cache"))
    ap.add_argument("--codec_ckpt", default=os.path.join(
        HERE, "ckpt", "wavtokenizer_large_unify_600_24k.ckpt"))
    ap.add_argument("--weights", default="ema", choices=["ema", "model"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--n_steps", type=int, default=32)
    ap.add_argument("--cfg_scale", type=float, default=2.0)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "diagnose.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    targs, cfgd = ck["args"], ck["cfg"]
    cfg = Cfg(**{k: v for k, v in cfgd.items() if k in Cfg.__dataclass_fields__})
    log(f"ckpt {args.ckpt}")
    log(f"  model {targs['model']} | steps {targs['steps']} | weights {args.weights}")
    log(f"  latent_std {cfg.latent_std} | noise_scale {cfg.denoiser_noise_scale}")

    val = LJCache(args.cache, "val", seed=targs["seed"])
    meta = val.meta
    coll = make_collate(targs["max_length"], (0.3, 0.3), targs["seed"], fixed_prompt=0.3)
    dl = DataLoader(val, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=coll)

    model = ELFTTS(targs["model"], meta["char_vocab_size"], meta["audio_vocab_size"],
                   dim=512, max_length=targs["max_length"],
                   cb_skip=targs.get("cb_skip", False)).to(device)
    model.load_state_dict(ck[args.weights])
    model.eval()

    codebook = load_codebook(args.codec_ckpt, device)
    cb = codebook / cfg.latent_std
    cb_sq = cb.pow(2).sum(1)

    res = {"ckpt": args.ckpt, "weights": args.weights, "train_args": targs}

    # ======================================================== A. teacher-forced denoiser
    log("")
    log("=== A. teacher-forced denoiser: can the model predict x0 from a true z_t? ===")
    log("    all distances are mean per-frame ||.||, in /latent_std units (so ||x0|| ~ sqrt(512)=22.6)")
    log(f"    {'t':>6} {'||x0_hat-x0||':>14} {'||z_t-x0||':>12} {'||x0||':>9}  {'beats z_t?':>10}  {'NN acc':>8} {'CE acc':>8}")
    ts_grid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99]
    A = []
    for tv in ts_grid:
        e_hat = e_z = e_0 = 0.0
        a_nn = a_ce = 0.0
        nb = 0
        for bi, b in enumerate(dl):
            if bi >= args.batches:
                break
            x0 = b["latent"].to(device) / cfg.latent_std
            code = b["code"].to(device); text = b["text"].to(device)
            prompt = b["prompt"].to(device)
            m = b["valid"].to(device) * (1 - prompt)
            B = x0.shape[0]
            t = torch.full((B,), tv, device=device)
            eps = torch.randn_like(x0)
            z = t.view(-1, 1, 1) * x0 + (1 - t.view(-1, 1, 1)) * eps * cfg.denoiser_noise_scale
            z = restore(z, x0, prompt)
            sc = torch.ones(B, device=device)
            out = model(z, restore(torch.zeros_like(z), x0, prompt), text, t,
                        self_cond_cfg_scale=sc, decoder_step_active=True)
            x_hat = out[0]
            logits = out[1]
            e_hat += rel(x_hat, x0, m); e_z += rel(z, x0, m)
            e_0 += rel(torch.zeros_like(x0), x0, m)
            a_nn += acc(nn_codes(z, cb, cb_sq), code, m)
            a_ce += acc(logits.argmax(-1), code, m)
            nb += 1
        e_hat, e_z, e_0 = e_hat/nb, e_z/nb, e_0/nb
        a_nn, a_ce = a_nn/nb, a_ce/nb
        log(f"    {tv:>6.2f} {e_hat:>14.3f} {e_z:>12.3f} {e_0:>9.3f}  "
            f"{'YES' if e_hat < e_z else 'no':>10}  {a_nn*100:>7.2f}% {a_ce*100:>7.2f}%")
        A.append({"t": tv, "x0hat_err": e_hat, "zt_err": e_z, "x0_norm": e_0,
                  "nn_acc": a_nn, "ce_acc": a_ce})
    res["A_teacher_forced"] = A

    # ======================================================== B. rollout
    log("")
    log("=== B. ODE rollout: does ||z_t - x0|| actually fall? ===")
    traj = []
    b = next(iter(dl))
    x0 = b["latent"].to(device) / cfg.latent_std
    code = b["code"].to(device); text = b["text"].to(device)
    prompt = b["prompt"].to(device)
    m = b["valid"].to(device) * (1 - prompt)
    B = x0.shape[0]
    z = torch.randn_like(x0) * cfg.denoiser_noise_scale
    z = restore(z, x0, prompt)
    x_pred = restore(torch.zeros_like(z), x0, prompt)
    ts = get_sampling_steps(args.n_steps, "uniform", cfg.denoiser_p_mean, cfg.denoiser_p_std, device)
    scb = torch.ones(B, device=device)
    zero_d = torch.zeros(B, device=device); one_d = torch.ones(B, device=device)
    log(f"    {'step':>5} {'t':>6} {'||z-x0||':>10} {'||x0_hat-x0||':>14}")
    for i in range(len(ts) - 1):
        t = ts[i].expand(B)
        out_c = model(z, x_pred, text, t, self_cond_cfg_scale=scb, drop_text=zero_d)
        v_c, x_c = net_to_v(out_c, z, t)
        v_c = restore(v_c, torch.zeros_like(z), prompt); x_c = restore(x_c, x0, prompt)
        if args.cfg_scale != 1.0:
            out_u = model(z, x_pred, text, t, self_cond_cfg_scale=scb, drop_text=one_d)
            v_u, _ = net_to_v(out_u, z, t)
            v_u = restore(v_u, torch.zeros_like(z), prompt)
            v = v_u + args.cfg_scale * (v_c - v_u)
        else:
            v = v_c
        z = restore(z + (ts[i + 1] - ts[i]) * v, x0, prompt)
        x_pred = x_c
        ez, ex = rel(z, x0, m), rel(x_c, x0, m)
        traj.append({"step": i, "t": float(ts[i]), "z_err": ez, "x0hat_err": ex})
        if i % 4 == 0 or i == len(ts) - 2:
            log(f"    {i:>5} {float(ts[i]):>6.3f} {ez:>10.3f} {ex:>14.3f}")
    res["B_rollout"] = traj

    # ======================================================== C. decode the final z
    log("")
    log("=== C. code accuracy on the tensors the sampler produces ===")
    t1 = torch.ones(B, device=device)
    _, logits_z = model(z, torch.zeros_like(z), text, t1,
                        self_cond_cfg_scale=scb, decoder_step_active=True)
    _, logits_xh = model(x_pred, torch.zeros_like(x_pred), text, t1,
                         self_cond_cfg_scale=scb, decoder_step_active=True)
    C = {
        "final_z_err": rel(z, x0, m), "final_x0hat_err": rel(x_pred, x0, m),
        "nn_on_final_z": acc(nn_codes(z, cb, cb_sq), code, m),
        "ce_on_final_z": acc(logits_z.argmax(-1), code, m),
        "nn_on_x0hat": acc(nn_codes(x_pred, cb, cb_sq), code, m),
        "ce_on_x0hat": acc(logits_xh.argmax(-1), code, m),
        "nn_on_true_x0": acc(nn_codes(x0, cb, cb_sq), code, m),
    }
    log(f"    ||final z  - x0||  {C['final_z_err']:.3f}")
    log(f"    ||final x0_hat-x0|| {C['final_x0hat_err']:.3f}")
    log(f"    nearest-codebook on true x0   {C['nn_on_true_x0']*100:6.2f}%   (sanity: should be ~100)")
    log(f"    nearest-codebook on final z   {C['nn_on_final_z']*100:6.2f}%")
    log(f"    CE head          on final z   {C['ce_on_final_z']*100:6.2f}%   <- what ELF actually does")
    log(f"    nearest-codebook on x0_hat    {C['nn_on_x0hat']*100:6.2f}%")
    log(f"    CE head          on x0_hat    {C['ce_on_x0hat']*100:6.2f}%   <- free fix if much higher")
    res["C_decode"] = C

    # ======================================================== D. CE head vs lambda
    log("")
    log("=== D. CE head over the distribution it was TRAINED on ===")
    log("    z_dec = lam*x0 + (1-lam)*eps ;  training draws lam ~ sigmoid(N(0.8,0.8)), mean 0.69")
    log(f"    {'lam':>6} {'||z-x0||':>10} {'NN acc':>9} {'CE acc':>9}  {'CE - NN':>8}")
    D = []
    for lv in [0.1, 0.3, 0.5, 0.69, 0.8, 0.9, 0.95, 0.99, 1.0]:
        a_nn = a_ce = e = 0.0
        nb = 0
        for bi, bb in enumerate(dl):
            if bi >= args.batches:
                break
            x0b = bb["latent"].to(device) / cfg.latent_std
            codeb = bb["code"].to(device); textb = bb["text"].to(device)
            pb = bb["prompt"].to(device)
            mb = bb["valid"].to(device) * (1 - pb)
            Bb = x0b.shape[0]
            zz = lv * x0b + (1 - lv) * torch.randn_like(x0b) * cfg.decoder_noise_scale
            tt = torch.ones(Bb, device=device)
            ss = torch.ones(Bb, device=device)
            _, lg = model(zz, torch.zeros_like(zz), textb, tt,
                          self_cond_cfg_scale=ss, decoder_step_active=True)
            a_nn += acc(nn_codes(zz, cb, cb_sq), codeb, mb)
            a_ce += acc(lg.argmax(-1), codeb, mb)
            e += rel(zz, x0b, mb)
            nb += 1
        a_nn, a_ce, e = a_nn/nb, a_ce/nb, e/nb
        log(f"    {lv:>6.2f} {e:>10.3f} {a_nn*100:>8.2f}% {a_ce*100:>8.2f}%  "
            f"{(a_ce-a_nn)*100:>+7.2f}")
        D.append({"lam": lv, "z_err": e, "nn_acc": a_nn, "ce_acc": a_ce})
    res["D_ce_vs_lambda"] = D

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    log("")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
