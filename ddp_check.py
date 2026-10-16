#!/usr/bin/env python
"""Prove the DDP gradient equals the single-GPU gradient on identical data.

The 4-GPU run logged a loss about 20% below the 1-GPU run at the same effective
batch. That could be harmless (rank 0 reports its own numerator over the global
denominator, which is an unbiased estimate of the global loss but not equal to
it) or it could be a wrong normalisation, which would silently change the
effective learning rate. Logging cannot tell those apart, so compare the thing
that actually trains the model: the gradient.

Run twice on the SAME fixed batch of 32 and compare:

    torchrun --standalone --nproc_per_node=1 ddp_check.py --batch 32
    torchrun --standalone --nproc_per_node=4 ddp_check.py --batch 8

Rank r takes rows [r*batch, (r+1)*batch) of the same 32-row batch, so the union
over ranks is exactly the single-GPU batch. If the normalisation is right the
reduced gradient matches to float tolerance.
"""
import argparse
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_unified as T
from unified import Cfg, UnifiedELF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_lj_uni")
    ap.add_argument("--batch", type=int, default=8, help="per-rank rows")
    ap.add_argument("--total", type=int, default=32, help="the effective batch")
    ap.add_argument("--max_length", type=int, default=560)
    ap.add_argument("--max_text", type=int, default=48)
    a = ap.parse_args()

    rank, world = T.RANK, T.WORLD
    if T.DDP:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(T.LOCAL_RANK)
    dev = torch.device("cuda", T.LOCAL_RANK if T.DDP else 0)
    assert a.batch * world == a.total, (
        f"per-rank {a.batch} x {world} ranks must equal the effective batch "
        f"{a.total}, otherwise the two runs see different data")

    ds = T.LibriCache(a.cache, "train")
    meta = ds.meta
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("t5-small")
    itos = {v: k for k, v in meta["stoi"].items()}
    coll = T.make_collate(a.max_length, 0.5, (0.0, 0.0), rng_seed=7,
                          max_text=a.max_text, itos=itos, tk=tk,
                          layout="hybrid",
                          whisper_rate=meta.get("whisper_rate", 0.0),
                          frame_rate=meta["frame_rate"])
    # one fixed batch of `total` rows, identical in both runs: no sampler, no
    # shuffle, the same collate seed
    dl = DataLoader(ds, batch_size=a.total, shuffle=False, num_workers=0,
                    collate_fn=coll)
    full = next(iter(dl))
    sl = slice(rank * a.batch, (rank + 1) * a.batch)
    batch = {k: (v[sl] if torch.is_tensor(v) and v.shape[0] == a.total else v)
             for k, v in full.items()}

    torch.manual_seed(0)
    wdim = meta.get("whisper_dim", 0)
    model = UnifiedELF("ELF-B", char_vocab=meta["char_vocab_size"],
                       max_length=a.max_length, whisper_dim=wdim,
                       use_cond=True).to(dev)
    if T.DDP:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[T.LOCAL_RANK], output_device=T.LOCAL_RANK)
    core = getattr(model, "module", model)
    ema = __import__("copy").deepcopy(core).eval()
    text_enc = T.TextEncoder(0.2).to(dev)
    cfg = Cfg(denoiser_p_mean=-1.5, decoder_p_mean=-0.5, decoder_prob=0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0)   # lr 0: never step
    mel_stats = (meta["latent_mean"], meta["latent_std"])

    # the same noise draw on every rank, so the only difference is the data
    torch.manual_seed(1234)
    m = T.train_step(model, text_enc, ema, batch, cfg, opt, dev, mel_stats)

    # train_step clips the gradient to norm 1.0 before opt.step(), so measuring
    # the norm here always returns exactly 1.0 and proves nothing. The pre-clip
    # norm is what train_step already returns.
    gn_post = 0.0
    sig = []
    for n, p in sorted(core.named_parameters()):
        if p.grad is None:
            continue
        gn_post += float(p.grad.float().pow(2).sum())
        if n in ("voice_out.weight", "elf.self_cond_proj.weight",
                 "whisper_proj.weight"):
            sig.append((n, float(p.grad.float().sum()),
                        float(p.grad.float().abs().mean())))
    gn_post = gn_post ** 0.5

    if rank == 0:
        print(f"world={world}  per-rank batch={a.batch}  effective={a.total}")
        print(f"  rank-0 loss            : {m['loss']:.6f}")
        print(f"  PRE-CLIP GRAD NORM     : {m['grad_norm']:.6f}   <-- 要比的是这个")
        print(f"  post-clip norm         : {gn_post:.6f}   (恒为 1.0，无信息)")
        print(f"  denom  本地 {m['denom_local']:.1f}   全局 {m['denom_global']:.1f}"
              f"   比值 {m['denom_global'] / max(m['denom_local'], 1e-9):.3f}"
              f"   <-- 4 卡时应约为 ranks 数（各 rank 行数相同）")
        for n, s, am in sig:
            print(f"  {n:30s} grad.sum {s:+.6e}  |grad|.mean {am:.6e}")
    if T.DDP:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
