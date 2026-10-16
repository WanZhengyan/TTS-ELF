#!/usr/bin/env python
"""At what latent scale was ELF-B-owt trained? Let its own denoiser answer.

ELF-B-owt's config declares the space it trained in:

    encoder_model_name: t5-small     text_encoder_dim 512, vocab 32100
    latent_mean: 0.0
    latent_std:  0.2

and a sibling codebase documents the convention these two obey:

    "Scalar normalisation by the table's own per-dimension std ... maps the
     typical row to norm 44.0 -- the same scale as the noise -- which is exactly
     the job ELF's latent_mean/latent_std do for T5. Stats are computed from the
     table itself, not a corpus."
                        -- UnderstandFlow/underflow/models/janus_text_space.py

    encode():  x0 = (E[ids] - latent_mean) / latent_std

So x0 is the embedding DIVIDED by latent_std, and the intent is per-dim std ~1.
But no t5 table has std 0.2: t5-small measures 23.19 and t5-v1_1-small 11.64.
The declared constant does not reproduce from either checkpoint, so the scale
has to be recovered from the weights instead of read off the config.

The denoiser can be asked directly. ELF is trained so that at t near 1

    z = t*x0 + (1-t)*eps*denoiser_noise_scale        (t=1 clean, t=0 noise)
    final_layer(z) ~= x0

and the output magnitude is pinned by the trained weights. Feed the SAME tokens
at many scales: at the training scale the prediction is self-consistent --
||x0_hat|| / ||x0|| ~= 1 and the relative error bottoms out. Away from it the
model's fixed output scale no longer matches the input, so the ratio drifts off 1
in a direction that says which way is wrong.

The decode head is reported alongside but is the weaker witness: it reads
direction far more than magnitude, so it keeps working 5x away from the truth.
"""
import os, sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from elf_tts.model import ELF_models

MAXLEN = 64
CKPT = os.path.join(HERE, "ckpt", "ELF-B-owt-torch", "checkpoint_95085")
DENOISER_NOISE_SCALE = 2.0      # from config.yml
SENTS = [
    "the quick brown fox jumps over the lazy dog",
    "she had your dark suit in greasy wash water all year",
    "printing in the only sense with which we are at present concerned",
    "a flow matching model generates text in a continuous embedding space",
    "in the beginning the universe was created this has made a lot of people",
    "we hold these truths to be self evident that all men are created equal",
]
# std(x0) values to test, i.e. what latent_std would have to be for each
TARGET_STDS = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 23.19, 34.0, 50.0, 100.0, 116.0, 200.0]


def log(*a):
    print(*a, flush=True)


@torch.no_grad()
def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoModel, AutoTokenizer

    tk = AutoTokenizer.from_pretrained("t5-small")
    E = AutoModel.from_pretrained("t5-small").shared.weight.data.float().to(dev)
    tbl_mu, tbl_sd = E.mean().item(), E.std().item()
    log(f"t5-small table {tuple(E.shape)}  mean {tbl_mu:+.5f}  std {tbl_sd:.5f}")

    # LibriSpeech transcripts are ALL CAPS and t5's sentencepiece fragments caps
    # badly (10 tokens upper vs 6 lower on one test line), so lowercase always.
    enc = tk(SENTS, return_tensors="pt", padding="max_length", max_length=MAXLEN)
    ids, am = enc.input_ids.to(dev), enc.attention_mask.to(dev)
    valid = am.bool()
    n_tok = int(valid.sum())
    B = ids.shape[0]
    log(f"{B} sentences, {n_tok} real tokens (padded to {MAXLEN})")

    model = ELF_models["ELF-B"](
        text_encoder_dim=512, max_length=MAXLEN, vocab_size=32100,
        n_input_blocks=2, bottleneck_dim=128, num_time_tokens=4,
        num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
    ).to(dev).eval()
    # n_input_blocks=2 is stock ELF: every tensor loads by shape, nothing partial,
    # nothing random. This is the released model, unmodified.
    src = torch.load(CKPT, map_location="cpu", weights_only=False)["ema_params1"]
    tgt = model.state_dict()
    bad = [k for k, v in src.items() if k not in tgt or tgt[k].shape != v.shape]
    for k, v in src.items():
        if k not in bad:
            tgt[k].copy_(v)
    model.load_state_dict(tgt)
    log(f"loaded {len(src)-len(bad)}/{len(src)} tensors"
        + (f"  NOT LOADED: {bad}" if bad else "  (clean)"))

    centred = F.embedding(ids, E) - tbl_mu          # (B, S, 512), ELF's latent_mean
    unit = centred / tbl_sd                         # per-dim std 1 in table units
    sc_cfg = torch.zeros(B, device=dev)
    g = torch.Generator(device=dev).manual_seed(0)

    def run(x0, t_val):
        t = torch.full((B,), t_val, device=dev)
        eps = torch.randn(x0.shape, device=dev, generator=g)
        z = t_val * x0 + (1.0 - t_val) * eps * DENOISER_NOISE_SCALE
        out, logits = model(torch.cat([z, torch.zeros_like(z)], dim=-1), t,
                            attention_mask=am, deterministic=True,
                            self_cond_cfg_scale=sc_cfg, decoder_step_active=True)
        return out, logits

    def masked_norm(a):
        return (a.float().pow(2) * valid.unsqueeze(-1)).sum().sqrt().item()

    log("")
    log("the denoiser's own verdict: is its prediction self-consistent at this scale?")
    log(f"(t=0.9, noise x{DENOISER_NOISE_SCALE})")
    log(f"{'std(x0)':>9}{'|xhat|/|x0|':>13}{'rel err':>10}   {'decode acc@t=1':>15}")
    sc_cfg = torch.zeros(ids.shape[0], device=dev)   # no self-cond CFG, but the
    # prefix tokens must still be present: the RoPE table budgets for them.
    rows = []
    for s in TARGET_STDS:
        x0 = unit * s
        out, _ = run(x0, 0.9)
        ratio = masked_norm(out) / max(masked_norm(x0), 1e-9)
        rel = masked_norm(out - x0) / max(masked_norm(x0), 1e-9)
        # decode head on the clean latent
        _, lg = run(x0, 1.0)
        acc = (((lg.argmax(-1) == ids) & valid).sum().item() / n_tok)
        rows.append((s, ratio, rel, acc))
        flag = ""
        log(f"{s:>9.2f}{ratio:>13.3f}{rel:>10.3f}   {acc*100:>14.2f}%{flag}")

    best_ratio = min(rows, key=lambda r: abs(r[1] - 1.0))
    best_rel = min(rows, key=lambda r: r[2])
    best_acc = max(rows, key=lambda r: r[3])
    log("")
    log(f"ratio closest to 1 : std(x0) = {best_ratio[0]:.2f}  "
        f"(ratio {best_ratio[1]:.3f}, rel err {best_ratio[2]:.3f})")
    log(f"lowest rel error   : std(x0) = {best_rel[0]:.2f}  (rel err {best_rel[2]:.3f})")
    log(f"best decode acc    : std(x0) = {best_acc[0]:.2f}  ({best_acc[3]*100:.2f}%)")
    log(f"chance acc = {100/32100:.4f}%")
    log("")
    if abs(best_rel[0] - best_ratio[0]) < 1e-6:
        log(f"-> both denoiser criteria agree: ELF trained on x0 with per-dim std "
            f"{best_rel[0]:.2f}.")
        log(f"   For t5-small (table std {tbl_sd:.3f}) that is")
        log(f"     x0 = (E - {tbl_mu:+.5f}) / {tbl_sd/best_rel[0]:.5f}")
        log(f"   i.e. an effective latent_std of {tbl_sd/best_rel[0]:.5f}, "
            f"not the config's 0.2.")
    else:
        log("-> the two denoiser criteria disagree; the scale is not sharply "
            "identified by this test.")


if __name__ == "__main__":
    main()
