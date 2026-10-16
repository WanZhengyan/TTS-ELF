#!/usr/bin/env python
"""Does the TTS head use the text? Measured directly, not read off the L2 curve.

The training L2 on mel is dominated by what needs no text at all -- the one
LJ voice, the spectral envelope, the prompt continuation -- so it can flatten
while the model still says the wrong words. This takes the SAME noisy mel at
fixed t and asks the model for x0_hat three times:

  true   the clip's own transcript
  shuf   another clip's transcript (stretched over the same frames)
  none   no text at all (latent, chars zeroed, drop_cond on: the CFG branch)

and reports the relative x0_hat error on the generated span. A model that
reads the text has true clearly below shuf; true ~= shuf means the words are
being ignored. Same noise for all three conditions, so the gap is paired.

  python diag_text_dep.py ckpt [ckpt ...]
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from unified import TTS, restore
from train_unified import LibriCache, TextEncoder, make_collate, build_targets, spk_cond
from eval_lj import load_model

TS = (0.1, 0.3, 0.5, 0.7, 0.9)


@torch.no_grad()
def run(path, n=16, seed=0, cache="cache_lj13k"):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ta = ck["args"]
    if ta.get("layout") != "hybrid":
        print(f"{path}: layout {ta.get('layout')} skipped"); return None
    model, meta = load_model(ck, dev)
    mk = ck.get("meta_keys") or meta     # older checkpoints predate meta_keys
    stats = (mk["latent_mean"], mk["latent_std"])
    te = TextEncoder(ta["text_latent_std"]).to(dev)
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("t5-small")
    itos = {v: k for k, v in meta["stoi"].items()}
    lj = LibriCache(os.path.join(HERE, cache), "all", dur_range=(4.0, 7.0))
    # each source standardised by its own stats, as in training (Tagged)
    sc_ = stats[1] / lj.meta["latent_std"]; sh_ = stats[0] - lj.meta["latent_mean"] * sc_
    idx = np.random.default_rng(seed).choice(len(lj), n, replace=False)
    base = [lj[int(i)] for i in idx]
    for it in base:
        it["mel"] = it["mel"].astype(np.float32) * sc_ + sh_
        if not model.spk_dim:
            it.pop("spk_emb", None)
    true = [{**it, "task": TTS} for it in base]
    shuf = [{**it, "chars": base[(k + 1) % n]["chars"], "t5": base[(k + 1) % n]["t5"],
             "task": TTS} for k, it in enumerate(base)]

    def coll():  # same seed -> same crops and prompt lengths for both batches
        return make_collate(ta["max_length"], 0.0, (0.2, 0.2), rng_seed=3,
                            max_text=ta["max_text"], itos=itos, tk=tk,
                            layout=ta["layout"], whisper_rate=0.0,
                            frame_rate=meta["frame_rate"])
    bt, bs = coll()(true), coll()(shuf)
    B = n
    ns = 2.0
    out = {}
    for t in TS:
        res = {}
        g = torch.Generator().manual_seed(seed + int(1000 * t))
        x0_v_ref = None
        for name, b in (("true", bt), ("shuf", bs), ("none", bt)):
            x0_t, x0_v = build_targets(b, te, dev, stats)
            if x0_v_ref is None:
                x0_v_ref = x0_v
                eps = torch.randn(x0_v.shape, generator=g).to(dev)
            chars = b["chars"].to(dev)
            dc = None
            if name == "none":
                x0_t = torch.zeros_like(x0_t); chars = torch.zeros_like(chars)
                dc = torch.ones(B, device=dev)
            va = b["v_act"].to(dev).unsqueeze(-1); tav = b["t_act"].to(dev).unsqueeze(-1)
            p_v, p_t = b["prompt_v"].to(dev), b["prompt_t"].to(dev)
            tt = torch.full((B,), t, device=dev)
            z_v = restore(t * x0_v + (1 - t) * eps * ns, x0_v, p_v) * va
            z_t = restore(x0_t, x0_t, p_t) * tav
            _, xh_v, _ = model(z_t, z_v, torch.zeros_like(z_t), torch.zeros_like(z_v), chars,
                               tt, b["task"].to(dev), attention_mask=b["valid"].to(dev),
                               self_cond_cfg_scale=torch.ones(B, device=dev), drop_cond=dc,
                               spk_emb=spk_cond(b, dev))
            m = (b["v_act"].to(dev) * (1 - p_v)).unsqueeze(-1)
            res[name] = float(((xh_v - x0_v_ref) ** 2 * m).sum() / ((x0_v_ref ** 2) * m).sum())
        out[t] = res
    print(f"\n{os.path.relpath(path, HERE)}  step {ck['step']}  on {cache}  "
          f"(data {os.path.basename(ta['cache'])}, p_s2t {ta['p_s2t']}, max_length {ta['max_length']})")
    print(f"  {'t':>4} {'true':>8} {'shuf':>8} {'none':>8}   shuf/true  none/true")
    for t, r in out.items():
        print(f"  {t:>4} {r['true']:8.4f} {r['shuf']:8.4f} {r['none']:8.4f}   "
              f"{r['shuf']/r['true']:8.3f}  {r['none']/r['true']:8.3f}")
    return out


if __name__ == "__main__":
    import json
    store = os.path.join(HERE, "out", "dual", "text_use.json")
    allr = json.load(open(store)) if os.path.exists(store) else {}
    cache = "cache_lj13k"
    args = sys.argv[1:]
    if args and args[0].startswith("--cache="):
        cache = args.pop(0).split("=", 1)[1]
    for p in args:
        r = run(p, cache=cache)
        if r is not None:
            ck = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
            ta = ck["args"]
            allr[os.path.relpath(p, HERE) + ("" if cache == "cache_lj13k" else f"@{cache}")] = {
                "step": ck["step"], "cache": os.path.basename(ta["cache"]),
                "p_s2t": ta["p_s2t"], "max_length": ta["max_length"],
                "batch": ta["batch"] * (4 if "dual" in p else 1),
                "res": {str(t): v for t, v in r.items()}}
            json.dump(allr, open(store, "w"), indent=1)
    print(f"wrote {store}")
