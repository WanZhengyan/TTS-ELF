#!/usr/bin/env python
"""Is ELF's text latent the t5 EMBEDDING TABLE or the t5 ENCODER OUTPUT?

probe_elf_scale.py found that ELF-B-owt's pretrained denoiser predicts x_hat0
no better than the trivial baseline on embedding-table lookups at any scale
(best relative error 0.883, where 1.000 = predicting nothing, and 1.786 at the
scale the config implies). Meanwhile its decode head recovers tokens at 93%.

That combination is diagnostic rather than contradictory. The decode head is
close to a nearest-embedding lookup, so it reads any vector that points at the
right row of E and is nearly scale-invariant. The denoiser is a regression and
fails only if the input distribution is wrong. So the latent is probably not a
table lookup at all -- and ELF's config calls the field

    encoder_model_name: t5-small

i.e. an ENCODER, whose contextual output is a different distribution entirely.

This measures both candidates under the config's own normalisation
(latent_mean 0.0, latent_std 0.2) and a scale sweep, and lets the denoiser's
relative error decide. rel err < 1.0 means it beats predicting nothing; the
correct latent should be far below 1.0 at t=0.9, where z is mostly clean.
"""
import os, sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from elf_tts.model import ELF_models

CKPT = os.path.join(HERE, "ckpt", "ELF-B-owt-torch", "checkpoint_95085")
MAXLEN = 64
SENTS = [
    "the quick brown fox jumps over the lazy dog",
    "she had your dark suit in greasy wash water all year",
    "printing in the only sense with which we are at present concerned",
    "a flow matching model generates text in a continuous embedding space",
    "the history of printing is in large measure the history of the book",
    "we hold these truths to be self evident that all men are created equal",
]


def log(*a):
    print(*a, flush=True)


@torch.no_grad()
def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer, T5EncoderModel, AutoModel

    tk = AutoTokenizer.from_pretrained("t5-small")
    enc_model = T5EncoderModel.from_pretrained("t5-small").to(dev).eval()
    E = AutoModel.from_pretrained("t5-small").shared.weight.data.float().to(dev)

    b = tk(SENTS, return_tensors="pt", padding="max_length", max_length=MAXLEN,
           truncation=True)
    ids, am = b.input_ids.to(dev), b.attention_mask.to(dev)
    valid = am.bool()
    n_tok = int(am.sum())

    table = F.embedding(ids, E)
    enc = enc_model(input_ids=ids, attention_mask=am).last_hidden_state.float()
    log(f"{len(SENTS)} sentences, {n_tok} real tokens")
    log(f"  table  lookup  std {table[valid].std():.4f}  mean {table[valid].mean():+.4f}")
    log(f"  encoder output std {enc[valid].std():.4f}  mean {enc[valid].mean():+.4f}")

    model = ELF_models["ELF-B"](
        text_encoder_dim=512, max_length=MAXLEN, vocab_size=32100,
        n_input_blocks=2, bottleneck_dim=128, num_time_tokens=4,
        num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
    ).to(dev).eval()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    tgt = model.state_dict()
    nload = 0
    for k, v in ck["ema_params1"].items():
        if k in tgt and tgt[k].shape == v.shape:
            tgt[k].copy_(v); nload += 1
    model.load_state_dict(tgt)
    log(f"loaded {nload}/{len(ck['ema_params1'])} tensors")

    sc_cfg = torch.zeros(len(SENTS), device=dev)

    def probe(x0, t_val=0.9, seed=0):
        g = torch.Generator(device=dev).manual_seed(seed)
        t = torch.full((x0.shape[0],), t_val, device=dev)
        eps = torch.randn(x0.shape, device=dev, generator=g) * 2.0
        z = t_val * x0 + (1.0 - t_val) * eps
        out, logits = model(torch.cat([z, torch.zeros_like(z)], dim=-1), t,
                            attention_mask=am, deterministic=True,
                            self_cond_cfg_scale=sc_cfg, decoder_step_active=True)
        xh = out[0] if isinstance(out, tuple) else out
        rel = (xh - x0)[valid].norm().item() / x0[valid].norm().item()
        acc = (((logits.argmax(-1) == ids) & valid).sum().item() / n_tok)
        return rel, acc, (xh[valid].norm() / x0[valid].norm()).item()

    log("")
    log("rel err 1.000 = predicting nothing (the trivial baseline).")
    log(f"{'latent':<18}{'target std':>11}{'|xhat|/|x0|':>13}"
        f"{'rel err':>10}{'decode acc':>12}")
    best = None
    for name, src in (("table lookup", table), ("encoder output", enc)):
        for std_target in (0.2, 1.0, 2.0, None):
            x0 = src.clone()
            if std_target is not None:
                x0 = (x0 - src[valid].mean()) / src[valid].std() * std_target
                tag = f"{name} -> {std_target}"
            else:
                tag = f"{name} raw"
            rel, acc, ratio = probe(x0)
            log(f"{tag:<18}{x0[valid].std():>11.3f}{ratio:>13.3f}"
                f"{rel:>10.4f}{acc*100:>11.1f}%")
            if best is None or rel < best[1]:
                best = (tag, rel, acc)

    log("")
    log(f"lowest rel err: {best[0]}  ({best[1]:.4f}, decode acc {best[2]*100:.1f}%)")
    if best[1] < 0.5:
        log("-> that is ELF's latent. The S2T half starts from a working denoiser")
        log("   AND a working decode head.")
    elif best[1] < 0.9:
        log("-> beats the trivial baseline but not convincingly. Neither")
        log("   candidate is ELF's latent; the input convention is still off.")
    else:
        log("-> neither candidate works. ELF init gives the S2T half nothing")
        log("   beyond the decode head, and the denoiser trains from scratch.")


if __name__ == "__main__":
    main()
