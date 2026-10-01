#!/usr/bin/env python
"""Does matching the sampling schedule to the training schedule fix the rollout?

No retraining. Same Test 2 checkpoint, two step grids:

  uniform       t = linspace(0,1,n+1)   -- queries t=0.97, 0.94, ... which
                                           training essentially never sampled
  logit_normal  t = sort(sigmoid(N(p_mean,p_std))) with 0 and 1 appended, i.e.
                ELF's own default: the denoiser is only ever queried at t values
                drawn from the training distribution, and t=1 is handled by the
                decoder branch, which IS trained at t=1.
"""
import argparse, json, os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train import (ELFTTS, LJCache, make_collate, Cfg, load_codebook,  # noqa: E402
                   get_sampling_steps, restore, net_to_v)
from torch.utils.data import DataLoader  # noqa: E402


def log(*a):
    print(*a, flush=True)


def nn_codes(z, cb, cb_sq):
    flat = z.reshape(-1, z.shape[-1])
    d = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ cb.t() + cb_sq[None, :]
    return d.argmin(1).view(z.shape[:-1])


def acc(pred, code, m):
    return (((pred == code).float() * m).sum() / m.sum().clamp(min=1)).item()


def rel(a, b, m):
    n = (a - b).norm(dim=-1)
    return ((n * m).sum() / m.sum().clamp(min=1)).item()


@torch.no_grad()
def rollout(model, b, cfg, device, cb, cb_sq, n_steps, schedule, cfg_scale, seed=0):
    torch.manual_seed(seed)
    x0 = b["latent"].to(device) / cfg.latent_std
    code = b["code"].to(device); text = b["text"].to(device)
    prompt = b["prompt"].to(device)
    m = b["valid"].to(device) * (1 - prompt)
    B = x0.shape[0]

    z = restore(torch.randn_like(x0) * cfg.denoiser_noise_scale, x0, prompt)
    x_pred = restore(torch.zeros_like(z), x0, prompt)
    ts = get_sampling_steps(n_steps, schedule, cfg.denoiser_p_mean, cfg.denoiser_p_std, device)
    scb = torch.ones(B, device=device)
    zd, od = torch.zeros(B, device=device), torch.ones(B, device=device)

    traj = []
    for i in range(len(ts) - 1):
        t = ts[i].expand(B)
        out_c = model(z, x_pred, text, t, self_cond_cfg_scale=scb, drop_text=zd)
        v_c, x_c = net_to_v(out_c, z, t)
        v_c = restore(v_c, torch.zeros_like(z), prompt); x_c = restore(x_c, x0, prompt)
        if cfg_scale != 1.0:
            out_u = model(z, x_pred, text, t, self_cond_cfg_scale=scb, drop_text=od)
            v_u, _ = net_to_v(out_u, z, t)
            v_u = restore(v_u, torch.zeros_like(z), prompt)
            v = v_u + cfg_scale * (v_c - v_u)
        else:
            v = v_c
        z = restore(z + (ts[i + 1] - ts[i]) * v, x0, prompt)
        x_pred = x_c
        traj.append({"i": i, "t": float(ts[i]), "z_err": rel(z, x0, m),
                     "x0hat_err": rel(x_c, x0, m)})

    t1 = torch.ones(B, device=device)
    _, lg_z = model(z, torch.zeros_like(z), text, t1, self_cond_cfg_scale=scb,
                    decoder_step_active=True)
    _, lg_x = model(x_pred, torch.zeros_like(x_pred), text, t1, self_cond_cfg_scale=scb,
                    decoder_step_active=True)
    return {
        "schedule": schedule, "n_steps": n_steps, "cfg_scale": cfg_scale,
        "t_grid": [float(v) for v in ts],
        "final_z_err": rel(z, x0, m), "final_x0hat_err": rel(x_pred, x0, m),
        "nn_on_z": acc(nn_codes(z, cb, cb_sq), code, m),
        "ce_on_z": acc(lg_z.argmax(-1), code, m),
        "nn_on_x0hat": acc(nn_codes(x_pred, cb, cb_sq), code, m),
        "ce_on_x0hat": acc(lg_x.argmax(-1), code, m),
        "traj": traj,
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "test2", "ckpt.pt"))
    ap.add_argument("--cache", default=os.path.join(HERE, "cache"))
    ap.add_argument("--codec_ckpt", default=os.path.join(
        HERE, "ckpt", "wavtokenizer_large_unify_600_24k.ckpt"))
    ap.add_argument("--weights", default="ema")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "sched_test.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    targs = ck["args"]
    cfg = Cfg(**{k: v for k, v in ck["cfg"].items() if k in Cfg.__dataclass_fields__})
    log(f"ckpt {args.ckpt} | {targs['model']} @ {targs['steps']} steps | weights {args.weights}")
    log(f"training t ~ sigmoid(N({cfg.denoiser_p_mean}, {cfg.denoiser_p_std}))")

    val = LJCache(args.cache, "val", seed=targs["seed"])
    meta = val.meta
    coll = make_collate(targs["max_length"], (0.3, 0.3), targs["seed"], fixed_prompt=0.3)
    dl = DataLoader(val, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=coll)
    b = next(iter(dl))

    model = ELFTTS(targs["model"], meta["char_vocab_size"], meta["audio_vocab_size"],
                   dim=512, max_length=targs["max_length"],
                   cb_skip=targs.get("cb_skip", False)).to(device)
    model.load_state_dict(ck[args.weights]); model.eval()
    cb = load_codebook(args.codec_ckpt, device) / cfg.latent_std
    cb_sq = cb.pow(2).sum(1)

    runs = []
    log("")
    log(f"{'schedule':>13} {'steps':>6} {'cfg':>5} | {'||z-x0||':>9} {'||x0h-x0||':>11} | "
        f"{'NN(z)':>7} {'CE(z)':>7} {'NN(x0h)':>8} {'CE(x0h)':>8}")
    log("-" * 94)
    for sched in ("uniform", "logit_normal"):
        for n in (16, 32, 64):
            for cg in (1.0, 2.0):
                r = rollout(model, b, cfg, device, cb, cb_sq, n, sched, cg)
                runs.append(r)
                log(f"{sched:>13} {n:>6} {cg:>5.1f} | {r['final_z_err']:>9.3f} "
                    f"{r['final_x0hat_err']:>11.3f} | {r['nn_on_z']*100:>6.2f}% "
                    f"{r['ce_on_z']*100:>6.2f}% {r['nn_on_x0hat']*100:>7.2f}% "
                    f"{r['ce_on_x0hat']*100:>7.2f}%")

    # show the two 32-step grids so the difference is visible
    log("")
    for sched in ("uniform", "logit_normal"):
        g = next(r["t_grid"] for r in runs if r["schedule"] == sched and r["n_steps"] == 32)
        log(f"{sched} 32-step grid (every 4th): " +
            " ".join(f"{v:.3f}" for v in g[::4]))
        log(f"    max t queried by the denoiser (excluding the t=1 endpoint): {max(g[:-1]):.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"runs": runs, "cfg": cfg.__dict__, "train_args": targs}, f, indent=2)
    log("")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
