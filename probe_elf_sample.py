#!/usr/bin/env python
"""Ask ELF-B-owt for its own latent scale by sampling from it.

Three probes narrowed the S2T target but did not settle it:

  table lookup, std 2    rel err 0.851   decode acc 68.2%
  encoder output, std 1  rel err 0.739   decode acc 45.9%
  encoder output, std 2  rel err 0.701   decode acc 77.6%
  encoder output, raw    rel err 1.661   decode acc  2.4%   (std 0.159)

1.000 is the trivial baseline, so none of these is a working denoiser. The one
hard clue is that the raw t5 ENCODER output has std 0.159, within a sixth of the
config's declared latent_std 0.2 -- the table lookup is 17.3, two orders out.
That points at encoder_output / latent_std, i.e. std ~1.

Rather than keep guessing the input, generate from the model. Unconditional
sampling needs no latent convention at all: start from noise, integrate the
model's own velocity field, and read off the result. If the samples decode to
coherent English then the checkpoint and our forward call are both fine and the
remaining error is a distribution mismatch in the probe inputs. And the std of
what it generates IS the scale it was trained in -- the model reports it.

ELF's convention: t=0 is noise, t=1 is clean, z = t*x0 + (1-t)*eps*noise_scale.
"""
import os, sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from elf_tts.model import ELF_models

CKPT = os.path.join(HERE, "ckpt", "ELF-B-owt-torch", "checkpoint_95085")
T_EPS = 0.05


def log(*a):
    print(*a, flush=True)


def net_to_v(out, z, t):
    x = out[0] if isinstance(out, tuple) else out
    return (x - z) / torch.clamp(1.0 - t.view(-1, 1, 1), min=T_EPS), x


@torch.no_grad()
def sample(model, B, L, D, dev, steps=64, noise_scale=2.0, sc_cfg_val=1.0,
           use_sc=True):
    """ELF's sampler: ODE on v, with self-conditioning fed from the last x_hat0."""
    z = torch.randn(B, L, D, device=dev) * noise_scale
    xp = torch.zeros_like(z)
    am = torch.ones(B, L, device=dev)
    sc_cfg = torch.full((B,), sc_cfg_val, device=dev)
    ts = torch.linspace(0.0, 1.0, steps + 1, device=dev)
    for i in range(steps):
        t = torch.full((B,), float(ts[i]), device=dev)
        out, _ = model(torch.cat([z, xp if use_sc else torch.zeros_like(z)],
                                 dim=-1), t, attention_mask=am,
                       deterministic=True, self_cond_cfg_scale=sc_cfg)
        v, xh = net_to_v(out, z, t)
        z = z + (float(ts[i + 1]) - float(ts[i])) * v
        xp = xh
    return z


@torch.no_grad()
def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("t5-small")

    for L in (128, 1024):
        model = ELF_models["ELF-B"](
            text_encoder_dim=512, max_length=L, vocab_size=32100,
            n_input_blocks=2, bottleneck_dim=128, num_time_tokens=4,
            num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
        ).to(dev).eval()
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        tgt = model.state_dict()
        n = 0
        for k, v in ck["ema_params1"].items():
            if k in tgt and tgt[k].shape == v.shape:
                tgt[k].copy_(v); n += 1
        model.load_state_dict(tgt)

        log("")
        log(f"================ max_length {L}  (loaded {n}/167) ================")
        for sc_cfg_val, use_sc in ((1.0, True), (2.0, True), (1.0, False)):
            z = sample(model, 2, L, 512, dev, steps=64, sc_cfg_val=sc_cfg_val,
                       use_sc=use_sc)
            # decode what it generated
            t1 = torch.ones(z.shape[0], device=dev)
            am = torch.ones(z.shape[0], L, device=dev)
            _, logits = model(torch.cat([z, torch.zeros_like(z)], dim=-1), t1,
                              attention_mask=am, deterministic=True,
                              self_cond_cfg_scale=torch.ones(z.shape[0], device=dev),
                              decoder_step_active=True)
            ids = logits.argmax(-1)
            txt = tk.decode(ids[0, :48], skip_special_tokens=True)
            log(f"sc_cfg {sc_cfg_val}  self_cond {use_sc}: "
                f"generated std {z.std().item():.4f}  "
                f"per-dim |z| {z.norm(dim=-1).mean().item():.2f}")
            log(f"    {txt[:220]}")
        del model
        torch.cuda.empty_cache()

    log("")
    log("The generated std is the scale the model was trained in. If the text")
    log("is coherent English, the checkpoint and the forward call are fine and")
    log("the probes' high error was a distribution mismatch in their inputs.")


if __name__ == "__main__":
    main()
