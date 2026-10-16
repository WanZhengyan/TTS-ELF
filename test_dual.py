#!/usr/bin/env python
"""Check the two-source data path before it gets 48 GPU-hours.

TTS rows must come from the TTS cache and S2T rows from the S2T caches; an S2T
clip's mel must reach build_targets standardised with its OWN statistics; a
batch with no S2T row must still carry the Whisper block (or DDP fails the
step); the sampler must split tasks by p_s2t, not by source size; and one real
train_step must run on a mixed batch. Runs on CPU.

  python test_dual.py [tts_cache] [s2t_cache,...] [eval_cache]
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import ConcatDataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_unified as T
from unified import Cfg, UnifiedELF, S2T, TTS

FAIL = []


def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"         {detail}")
    if not ok:
        FAIL.append(name)


def main():
    tts_c = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "cache_lj13k")
    s2t_cs = (sys.argv[2] if len(sys.argv) > 2 else
              os.path.join(HERE, "cache_libri_tc100_lv3")).split(",")
    ev_c = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "cache_libri_test_clean_lv3")
    torch.manual_seed(0)
    tb = T.LibriCache(tts_c, "all")
    meta = tb.meta
    ref = (meta["latent_mean"], meta["latent_std"])
    sbs = [T.LibriCache(c, "all") for c in s2t_cs]
    wdim, wrate = sbs[0].meta["whisper_dim"], sbs[0].meta["whisper_rate"]
    tds = T.Tagged(tb, TTS, ref)
    sds = [T.Tagged(b, S2T, ref) for b in sbs]
    ds = ConcatDataset([tds] + sds)
    n_tts, n_s2t = len(tds), sum(len(d) for d in sds)
    print(f"TTS {os.path.basename(tts_c)} {n_tts} clips | S2T "
          f"{[os.path.basename(c) for c in s2t_cs]} {n_s2t} clips | "
          f"whisper {wdim}-d @ {wrate} Hz")

    print("\n1. per-source normalisation lands on each source's own stats")
    for name, d, base in [("TTS", tds, tb)] + [("S2T", d, b) for d, b in zip(sds, sbs)]:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(d), 200, replace=False)
        x = np.concatenate([(d[int(i)]["mel"] - ref[0]) / ref[1] for i in idx])
        raw = np.concatenate([base[int(i)]["mel"].astype(np.float32) for i in idx])
        check(f"{name} {os.path.basename(base.meta.get('split', ''))} standardised",
              abs(x.mean()) < 0.1 and abs(x.std() - 1) < 0.1,
              f"after build_targets' (x-{ref[0]:.3f})/{ref[1]:.3f}: mean {x.mean():+.3f} "
              f"std {x.std():.3f} (raw mean {raw.mean():+.3f} std {raw.std():.3f})")

    print("\n2. sampler splits by p_s2t, not by source size, and per rank")
    s = T.TaskSampler(n_tts, n_s2t, 0.5, seed=[0, 0, 17])
    it = iter(s)
    draws = np.array([next(it) for _ in range(20000)])
    frac = (draws >= n_tts).mean()
    check("S2T fraction ~0.5", abs(frac - 0.5) < 0.02,
          f"{frac:.4f} of 20000 draws (shuffling the union would give "
          f"{n_s2t / (n_tts + n_s2t):.3f})")
    cover = len(np.unique(draws[draws < n_tts])) / min(n_tts, (draws < n_tts).sum())
    check("TTS draws spread over the TTS cache", cover > 0.4,
          f"{len(np.unique(draws[draws < n_tts]))} distinct TTS clips in "
          f"{(draws < n_tts).sum()} draws")
    s1 = T.TaskSampler(n_tts, n_s2t, 0.5, seed=[1, 0, 17])
    s0r = T.TaskSampler(n_tts, n_s2t, 0.5, seed=[0, 40000, 17])
    a0 = [next(iter(s)) for _ in range(1)]
    i0, i1, i2 = iter(s), iter(s1), iter(s0r)
    x0 = [next(i0) for _ in range(64)]; x1 = [next(i1) for _ in range(64)]
    x2 = [next(i2) for _ in range(64)]
    check("ranks and resumed segments draw different streams",
          x0 != x1 and x0 != x2, f"rank0 {x0[:4]} rank1 {x1[:4]} rank0@40k {x2[:4]}")

    print("\n3. a mixed batch: task follows the source, blocks gated by task")
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("t5-small")
    itos = {v: k for k, v in meta["stoi"].items()}
    coll = T.make_collate(1648, 0.5, (0.0, 0.3), max_text=112, itos=itos, tk=tk,
                          layout="hybrid", whisper_rate=wrate,
                          frame_rate=meta["frame_rate"], whisper_dim=wdim)
    picks = [0, 1, 2, n_tts, n_tts + 1, n_tts + 2, 3, n_tts + 3]
    items = [ds[i] for i in picks]
    b = coll(items)
    want = [TTS, TTS, TTS, S2T, S2T, S2T, TTS, S2T]
    check("task per row == source", b["task"].tolist() == want,
          f"task {b['task'].tolist()} uids {[u[:12] for u in b['uid']]}")
    s2t_rows = b["task"] == S2T
    vf = b["v_act"] > 0
    wm = b["w_mask"] > 0
    check("Whisper block present and only on S2T audio frames",
          "wh" in b and b["wh"].shape[-1] == wdim
          and bool((wm == (vf & s2t_rows[:, None])).all()),
          f"wh {tuple(b['wh'].shape)}; w_mask on S2T rows "
          f"{wm[s2t_rows].float().sum().item():.0f} = their frames "
          f"{vf[s2t_rows].float().sum().item():.0f}; on TTS rows "
          f"{wm[~s2t_rows].float().sum().item():.0f}")
    nz = b["wh"].abs().sum((1, 2))
    check("Whisper values zero on TTS rows, nonzero on S2T rows",
          bool((nz[~s2t_rows] == 0).all() and (nz[s2t_rows] > 0).all()),
          f"|wh| per row {[round(v, 1) for v in nz.tolist()]}")
    ch = (b["chars"] > 0).sum(1)
    check("char block only on TTS rows (no S2T answer leak)",
          bool((ch[s2t_rows] == 0).all() and (ch[~s2t_rows] > 0).all()),
          f"nonzero chars per row {ch.tolist()}")
    pv = b["prompt_v"].sum(1); va = b["v_act"].sum(1)
    check("S2T rows: all audio is condition; TTS rows: a prefix at most 30%",
          bool((pv[s2t_rows] == va[s2t_rows]).all()
               and (pv[~s2t_rows] <= 0.3 * va[~s2t_rows] + 1).all()),
          f"prompt_v/v_act {[f'{int(p)}/{int(v)}' for p, v in zip(pv, va)]}")

    print("\n4. a batch with NO S2T row still carries the Whisper block")
    bt = coll([ds[i] for i in range(8)])
    check("all-TTS batch has wh (zeros) so whisper_proj stays in the graph",
          "wh" in bt and float(bt["wh"].abs().sum()) == 0.0
          and float(bt["w_mask"].sum()) == 0.0,
          f"keys has wh: {'wh' in bt}; |wh| {float(bt['wh'].abs().sum()) if 'wh' in bt else None}")

    print("\n5. the eval set alternates the two tasks")
    ev_t = T.Tagged(T.LibriCache(tts_c, "all", dur_range=(4.0, 7.0)), TTS, ref)
    ev_s = T.Tagged(T.LibriCache(ev_c, "all"), S2T, ref)
    vs = T.Interleave(ev_t, ev_s)
    be = coll([vs[k] for k in range(8)])
    check("eval batch is TTS,S2T,TTS,S2T,...",
          be["task"].tolist() == [TTS, S2T] * 4,
          f"{be['task'].tolist()}; LJ 4-7 s pool {len(ev_t)}, test-clean {len(ev_s)}")

    print("\n6. one real train_step on the mixed batch (CPU, random init)")
    model = UnifiedELF("ELF-B", char_vocab=meta["char_vocab_size"], max_length=1648,
                       whisper_dim=wdim, use_cond=True, spk_dim=0)
    import copy
    ema = copy.deepcopy(model).eval()
    te = T.TextEncoder(0.2)
    cfg = Cfg(decoder_p_mean=-0.5, decoder_prob=0.5, audio_drop_prob=0.1,
              text_drop_prob=0.2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    small = coll([ds[0], ds[n_tts]])
    before = model.whisper_proj.weight.detach().clone()
    m = T.train_step(model, te, ema, small, cfg, opt, torch.device("cpu"), ref)
    moved = float((model.whisper_proj.weight - before).abs().max())
    check("train_step runs, loss finite, whisper_proj updated",
          np.isfinite(m["loss"]) and moved > 0,
          f"loss {m['loss']:.4f} gnorm {m['grad_norm']:.2f} frac_s2t {m['frac_s2t']:.2f} "
          f"|dW whisper_proj| {moved:.2e}")
    allt = coll([ds[0], ds[1]])
    m2 = T.train_step(model, te, ema, allt, cfg, opt, torch.device("cpu"), ref)
    check("train_step on an all-TTS batch runs", np.isfinite(m2["loss"]),
          f"loss {m2['loss']:.4f} frac_s2t {m2['frac_s2t']:.2f}")

    print(f"\n{'ALL PASS' if not FAIL else 'FAILED: ' + ', '.join(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
