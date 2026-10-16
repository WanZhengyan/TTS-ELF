#!/usr/bin/env python
"""How much of S2T's 17.20% WER actually comes from the flow matching?

The token logits are produced by a SEPARATE head on the transformer hidden
state (elf_tts/model.py:266: hidden @ unembed_kernel), trained with cross
entropy, not by the flow-matching output (final_layer). So the rollout's only
job is to put the hidden state somewhere the classifier can read. If 1 step
scores like 128 steps, our S2T is a Whisper-conditioned classifier and the
flow matching on the text latent is close to decorative -- which would change
what we can claim about the method.
"""
import argparse, os, sys
import torch
from torch.utils.data import DataLoader
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from unified import Cfg, UnifiedELF
from train_unified import LibriCache, TextEncoder, make_collate
from eval_protocol import s2t_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="libri_tc100_uni_ddp")
    ap.add_argument("--cache", default=os.path.join(HERE, "cache_libri_test_clean"))
    ap.add_argument("--steps", default="1,2,4,8,32,128")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(os.path.join(HERE, "runs", a.run, "ckpt.pt"),
                    map_location="cpu", weights_only=False)
    ta = ck["args"]
    ds = LibriCache(a.cache, "all"); meta = ds.meta
    itos = {v: k for k, v in meta["stoi"].items()}
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("t5-small")
    cfg = Cfg(denoiser_p_mean=ta["denoiser_p_mean"],
              decoder_p_mean=ta["decoder_p_mean"])
    wdim = meta.get("whisper_dim", 0) if ta.get("whisper") else 0
    wrate = meta.get("whisper_rate", 0.0) if ta.get("whisper") else 0.0
    model = UnifiedELF(ta["model"], char_vocab=meta["char_vocab_size"],
                       max_length=ta["max_length"], whisper_dim=wdim,
                       use_cond=not ta.get("no_cond", 0),
                       spk_dim=(meta.get("spk_emb_dim", 0)
                                if ta.get("spk") else 0)).to(dev)
    model.load_state_dict(ck["ema"]); model.eval()
    te = TextEncoder(ta["text_latent_std"]).to(dev)
    coll = make_collate(ta["max_length"], 1.0, (0.0, 0.0), rng_seed=7,
                        max_text=ta["max_text"], itos=itos, tk=tk,
                        layout=ta["layout"], whisper_rate=wrate,
                        frame_rate=meta["frame_rate"])
    print(f"  {a.run} @ step {ck['step']}   n={a.n}")
    print(f"  {'步数':>6}{'WER':>9}{'token 正确率':>14}")
    for st in [int(x) for x in a.steps.split(",")]:
        torch.manual_seed(0)
        dl = DataLoader(ds, batch_size=a.batch, collate_fn=coll)
        r = s2t_eval(model, te, dl, cfg, dev, (meta["latent_mean"],
                     meta["latent_std"]), tk, ta["layout"], st, 1.0, a.n)
        print(f"  {st:6d}{r['wer']:8.2f}%{r['tok_acc']*100:13.2f}%")
    print()
    print("  若 1 步 ≈ 128 步，则流匹配对 S2T 的贡献接近于零，")
    print("  17.20% 实际来自「Whisper 特征 + 交叉熵分类头」。")


if __name__ == "__main__":
    main()
