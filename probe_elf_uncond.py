#!/usr/bin/env python
"""Run ELF-B-owt the way it was trained: unconditional text generation.

Two questions at once.

1. Is our vendored ELF faithful? ELF-B-owt is a language model trained on
   OpenWebText. If elf_tts/model.py reproduces it and the checkpoint loads, the
   rollout produces English. If anything is off -- RoPE positions, the prefix
   token layout, the two-head wiring -- it produces token salad. Nothing should
   be built on the S2T half until this passes.

2. What latent scale did it train in? config.yml says latent_std 0.2, but no t5
   table reproduces that (t5-small 23.19, t5-v1_1-small 11.64), and the denoiser
   cannot be asked directly: denoiser_p_mean -1.5 puts its training mass at
   t ~ 0.18, where the correct x0_hat is near the conditional MEAN, so a
   self-consistency test at high t measures nothing. Generation sidesteps this.
   x0 never enters a rollout -- only the noise does -- so whatever scale the
   model lands on at t=1 IS the scale it was trained to produce.
"""
import math, os, sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from elf_tts.model import ELF_models

MAXLEN = 64
CKPT = os.path.join(HERE, "ckpt", "ELF-B-owt-torch", "checkpoint_95085")
# all from ckpt/ELF-B-owt-torch/config.yml
DENOISER_NOISE_SCALE, P_MEAN, P_STD, T_EPS = 2.0, -1.5, 0.8, 0.05


def log(*a):
    print(*a, flush=True)


def steps(n, dev):
    s = torch.sort(torch.sigmoid(torch.randn(n - 1, device=dev) * P_STD + P_MEAN)).values
    return torch.cat([torch.zeros(1, device=dev), s, torch.ones(1, device=dev)])


@torch.no_grad()
def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("t5-small")

    model = ELF_models["ELF-B"](
        text_encoder_dim=512, max_length=MAXLEN, vocab_size=32100,
        n_input_blocks=2, bottleneck_dim=128, num_time_tokens=4,
        num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
    ).to(dev).eval()
    src = torch.load(CKPT, map_location="cpu", weights_only=False)["ema_params1"]
    tgt = model.state_dict()
    bad = [k for k, v in src.items() if k not in tgt or tgt[k].shape != v.shape]
    for k, v in src.items():
        if k not in bad:
            tgt[k].copy_(v)
    model.load_state_dict(tgt)
    log(f"loaded {len(src)-len(bad)}/{len(src)} tensors"
        + (f"  NOT LOADED {bad}" if bad else "  (clean)"))

    B, N = 4, 64
    torch.manual_seed(0)
    am = torch.ones(B, MAXLEN, dtype=torch.long, device=dev)
    sc_cfg = torch.zeros(B, device=dev)

    for use_sc in (True, False):
        z = torch.randn(B, MAXLEN, 512, device=dev) * DENOISER_NOISE_SCALE
        ts = steps(N, dev)
        x_hat = torch.zeros_like(z)
        traj = []
        for i in range(N):
            t_cur = ts[i]
            t = torch.full((B,), float(t_cur), device=dev)
            sc = x_hat if use_sc else torch.zeros_like(z)
            out, _ = model(torch.cat([z, sc], dim=-1), t, attention_mask=am,
                           deterministic=True, self_cond_cfg_scale=sc_cfg,
                           decoder_step_active=None)
            x_hat = out.float()
            v = (x_hat - z) / max(float(1.0 - t_cur), T_EPS)
            z = z + (float(ts[i + 1]) - float(t_cur)) * v
            if i in (0, N // 4, N // 2, 3 * N // 4, N - 1):
                traj.append((float(t_cur), z.std().item(), x_hat.std().item()))

        t1 = torch.ones(B, device=dev)
        _, logits = model(torch.cat([z, x_hat if use_sc else torch.zeros_like(z)], dim=-1),
                          t1, attention_mask=am, deterministic=True,
                          self_cond_cfg_scale=sc_cfg, decoder_step_active=True)
        ids = logits.argmax(-1)

        log("")
        log(f"=== self-conditioning {'ON' if use_sc else 'OFF'}, {N} steps ===")
        log(f"  {'t':>6}{'std(z)':>10}{'std(x_hat)':>12}")
        for t_, zs, xs in traj:
            log(f"  {t_:>6.3f}{zs:>10.3f}{xs:>12.3f}")
        log(f"  -> ELF generates x0 at per-dim std {z.std().item():.3f}  "
            f"(noise entered at {DENOISER_NOISE_SCALE:.1f})")
        for b in range(B):
            txt = tk.decode(ids[b], skip_special_tokens=True)
            log(f"  [{b}] {txt[:150]}")


if __name__ == "__main__":
    main()
