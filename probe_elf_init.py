#!/usr/bin/env python
"""Does ELF's pretrained decode head read OUR T5 latents? Decide the scaling.

ELF-B-owt's config declares the latent space it was trained in:

    encoder_model_name: t5-small     ->  text_encoder_dim 512, vocab 32100
    latent_mean: 0.0
    latent_std:  0.2

but it does not say which statistic 0.2 refers to. t5-small's raw embedding
matrix has std 23.19 and mean +0.122, so the data was rescaled by ~1/116
somewhere, and "latent_std 0.2" could describe

    A  E / E.std() * 0.2                 matrix std, not centred
    B  (E - E.mean()) / E.std() * 0.2    matrix std, centred
    C  per-dim standardised, then * 0.2
    D  E * 0.2                           (control: wrong by ~116x)
    E  E                                 (control: raw)

Guessing is not necessary. ELF's decode head maps a clean latent back to token
ids, so at t=1 (ELF's convention: t=1 clean, t=0 noise) the CORRECT scaling must
recover the input tokens at near-100% accuracy and every wrong one must sit near
chance. The head is a fixed, pretrained function -- it votes.

This also tests the S2T half of the unified model before any of it is built:
if the pretrained head cannot read our latents, ELF init buys the S2T direction
nothing and we would be training it from scratch, like the mel run.
"""
import os, sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from elf_tts.model import ELF_models

MAXLEN = 64
CKPT = os.path.join(HERE, "ckpt", "ELF-B-owt-torch", "checkpoint_95085")
SENTS = [
    "the quick brown fox jumps over the lazy dog",
    "she had your dark suit in greasy wash water all year",
    "printing in the only sense with which we are at present concerned",
    "a flow matching model generates text in a continuous embedding space",
]


def log(*a):
    print(*a, flush=True)


@torch.no_grad()
def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoModel, AutoTokenizer

    tk = AutoTokenizer.from_pretrained("t5-small")
    E = AutoModel.from_pretrained("t5-small").shared.weight.data.float()
    log(f"t5-small embedding {tuple(E.shape)}  mean {E.mean():+.5f}  std {E.std():.5f}")

    # LibriSpeech transcripts are ALL CAPS and t5's sentencepiece fragments caps
    # badly (19 chars -> 10 tokens upper vs 6 lower), so everything is lowercased.
    # ELF's RoPE table is built for exactly max_length + prefix positions, so
    # the batch has to be padded to max_length. pos = arange(L)/L*L = arange(L),
    # i.e. plain RoPE, so a shorter max_length is position-identical to ELF's
    # 1024 -- it only shrinks the cache.
    enc = tk([s for s in SENTS], return_tensors="pt",
             padding="max_length", max_length=MAXLEN)
    ids, am = enc.input_ids.to(dev), enc.attention_mask.to(dev)
    log(f"{len(SENTS)} sentences -> ids {tuple(ids.shape)} (padded to {MAXLEN}), "
        f"{int(am.sum())} real tokens")

    Ed = E.to(dev)
    mu, sd = Ed.mean(), Ed.std()
    per_dim_mu, per_dim_sd = Ed.mean(0, keepdim=True), Ed.std(0, keepdim=True)
    scalings = {
        "A  E/std*0.2        ": lambda e: e / sd * 0.2,
        "B  (E-mean)/std*0.2 ": lambda e: (e - mu) / sd * 0.2,
        "C  per-dim std *0.2 ": lambda e: (e - per_dim_mu) / per_dim_sd * 0.2,
        "D  E*0.2  (control) ": lambda e: e * 0.2,
        "E  E raw  (control) ": lambda e: e,
    }

    model = ELF_models["ELF-B"](
        text_encoder_dim=512, max_length=MAXLEN, vocab_size=32100,
        n_input_blocks=2, bottleneck_dim=128, num_time_tokens=4,
        num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
    ).to(dev).eval()
    # n_input_blocks=2 is stock ELF: every tensor loads by shape, nothing partial,
    # nothing random. This is the released model, unmodified.
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    src = ck["ema_params1"]
    tgt = model.state_dict()
    ok = bad = 0
    for k, v in src.items():
        if k in tgt and tgt[k].shape == v.shape:
            tgt[k].copy_(v); ok += 1
        else:
            log(f"  NOT LOADED {k} {tuple(v.shape)} vs "
                f"{tuple(tgt[k].shape) if k in tgt else 'absent'}"); bad += 1
    model.load_state_dict(tgt)
    log(f"loaded {ok} tensors, {bad} not loaded  "
        f"({'clean' if bad == 0 else 'INCOMPLETE'})")

    raw = F.embedding(ids, Ed)                      # (B, S, 512)
    sc_cfg = torch.zeros(ids.shape[0], device=dev)  # no self-cond CFG
    valid = am.bool()
    n_tok = int(valid.sum())
    chance = 1.0 / 32100

    log("")
    log("decode-head accuracy on the CLEAN latent (t=1, ELF convention)")
    log(f"{'scaling':<22}{'std(x0)':>9}{'acc@t=1':>10}{'acc@t=.9':>10}"
        f"{'acc@t=.69':>11}")
    best = (None, -1.0)
    for name, fn in scalings.items():
        x0 = fn(raw)
        row = [f"{name:<22}{x0.std().item():>9.4f}"]
        accs = {}
        for t_val in (1.0, 0.9, 0.69):      # 0.69 = sigmoid(decoder_p_mean 0.8)
            t = torch.full((ids.shape[0],), t_val, device=dev)
            # ELF's path: z = t*x0 + (1-t)*eps*decoder_noise_scale; at t=1, z = x0.
            eps = torch.randn_like(x0) * 5.0          # decoder_noise_scale
            z = t_val * x0 + (1.0 - t_val) * eps
            sc = torch.zeros_like(z)                  # no self-conditioning
            _, logits = model(torch.cat([z, sc], dim=-1), t,
                              attention_mask=am, deterministic=True,
                              self_cond_cfg_scale=sc_cfg,
                              decoder_step_active=True)
            pred = logits.argmax(-1)
            acc = ((pred == ids) & valid).sum().item() / n_tok
            accs[t_val] = acc
            row.append(f"{acc*100:>9.2f}%")
        log("".join(row))
        if accs[1.0] > best[1]:
            best = (name, accs[1.0])

    log("")
    log(f"chance = {chance*100:.4f}%  ({n_tok} tokens, vocab 32100)")
    log(f"winner: {best[0].strip()}  at {best[1]*100:.2f}%")
    if best[1] > 0.9:
        log("-> ELF's pretrained decode head READS our T5 latents. The S2T half")
        log("   of the unified model starts from a working text decoder.")
    elif best[1] > 0.1:
        log("-> partial. Scaling is in the right region but not exact.")
    else:
        log("-> NO scaling works. ELF init buys the S2T direction nothing;")
        log("   it would train from scratch, like the mel run.")

    # What the winner actually decodes, as a sanity check on the accuracy number.
    name, fn = best[0], scalings[best[0]]
    x0 = fn(raw)
    t = torch.ones(ids.shape[0], device=dev)
    _, logits = model(torch.cat([x0, torch.zeros_like(x0)], dim=-1), t,
                      attention_mask=am, deterministic=True,
                      self_cond_cfg_scale=sc_cfg, decoder_step_active=True)
    log("")
    log("round-trip under the winning scaling:")
    for i in range(len(SENTS)):
        n = int(am[i].sum())
        got = tk.decode(logits[i, :n].argmax(-1), skip_special_tokens=True)
        log(f"  in   {SENTS[i]}")
        log(f"  out  {got}")


if __name__ == "__main__":
    main()
