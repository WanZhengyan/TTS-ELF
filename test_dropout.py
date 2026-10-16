#!/usr/bin/env python
"""Prove the condition dropout reaches the model's actual input.

Both drops used to zero x0 AFTER z had been built from it, so the model's main
input kept the full condition and only the self-conditioning copies went to
zero. Reordering the code is not evidence the fix works -- what the model
RECEIVES is. This hooks UnifiedELF.forward, runs one real train_step with the
drop probability at 1.0, and checks the tensors the model was handed.
"""
import os
import sys

import torch
from torch.utils.data import DataLoader

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
    torch.manual_seed(0)
    ds = T.LibriCache(os.path.join(HERE, "cache_lj_uni"), "train")
    meta = ds.meta
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("t5-small")
    itos = {v: k for k, v in meta["stoi"].items()}
    coll = T.make_collate(560, 0.5, (0.0, 0.3), rng_seed=3, max_text=48,
                          itos=itos, tk=tk, layout="hybrid",
                          whisper_rate=meta.get("whisper_rate", 0.0),
                          frame_rate=meta["frame_rate"])
    b = next(iter(DataLoader(ds, batch_size=8, collate_fn=coll)))
    task = b["task"]
    n_tts, n_s2t = int((task == TTS).sum()), int((task == S2T).sum())
    print(f"  batch: {n_tts} TTS rows, {n_s2t} S2T rows")
    assert n_tts and n_s2t, "need both tasks in the batch"

    dev = torch.device("cpu")
    model = UnifiedELF("ELF-B", char_vocab=meta["char_vocab_size"],
                       max_length=560, whisper_dim=meta.get("whisper_dim", 0),
                       use_cond=True).to(dev)
    ema = __import__("copy").deepcopy(model).eval()
    te = T.TextEncoder(0.2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0)
    mel_stats = (meta["latent_mean"], meta["latent_std"])
    is_tts = (task == TTS)
    is_s2t = (task == S2T)

    for name, cfg_kw in (("text_drop_prob=1.0", dict(text_drop_prob=1.0)),
                         ("audio_drop_prob=1.0", dict(audio_drop_prob=1.0)),
                         ("no drop (control)", dict())):
        seen = []
        orig = model.forward

        def hook(z_t, z_v, sc_t, sc_v, chars, t, task_id, **kw):
            seen.append(dict(z_t=z_t.detach().clone(), z_v=z_v.detach().clone(),
                             sc_t=sc_t.detach().clone(), chars=chars.clone(),
                             ac=(kw.get("audio_cond").detach().clone()
                                 if kw.get("audio_cond") is not None else None)))
            return orig(z_t, z_v, sc_t, sc_v, chars, t, task_id, **kw)

        model.forward = hook
        cfg = Cfg(denoiser_p_mean=-1.5, decoder_p_mean=-0.5, decoder_prob=0.0,
                  label_drop_prob=0.0, self_cond_prob=0.0, **cfg_kw)
        torch.manual_seed(0)
        T.train_step(model, te, ema, b, cfg, opt, dev, mel_stats)
        model.forward = orig
        x = seen[-1]          # the forward that feeds the loss

        print()
        print(f"  --- {name} ---")
        zt_tts = float(x["z_t"][is_tts].abs().sum())
        ch_tts = int((x["chars"][is_tts] != 0).sum())
        zv_s2t = float(x["z_v"][is_s2t].abs().sum())
        ac_s2t = (float(x["ac"][is_s2t].abs().sum())
                  if x["ac"] is not None else 0.0)
        zt_s2t = float(x["z_t"][is_s2t].abs().sum())
        zv_tts = float(x["z_v"][is_tts].abs().sum())
        if "text" in name:
            check("TTS 行：模型收到的 z_T 全为 0（文本真的被丢了）",
                  zt_tts == 0.0, f"|z_T| 之和 = {zt_tts:.4f}")
            check("TTS 行：chars 全为 0", ch_tts == 0, f"非零字符 {ch_tts}")
            check("TTS 行：z_V（生成目标）未被误伤",
                  zv_tts > 0.0, f"|z_V| 之和 = {zv_tts:.2f}")
            check("S2T 行：z_T 未被误伤（text_drop 只作用于 TTS）",
                  zt_s2t > 0.0, f"|z_T| 之和 = {zt_s2t:.2f}")
        elif "audio" in name:
            check("S2T 行：模型收到的 z_V 全为 0（mel 真的被丢了）",
                  zv_s2t == 0.0, f"|z_V| 之和 = {zv_s2t:.4f}")
            check("S2T 行：Whisper 特征全为 0", ac_s2t == 0.0,
                  f"|ac| 之和 = {ac_s2t:.4f}")
            check("TTS 行：z_V 未被误伤（audio_drop 只作用于 S2T）",
                  zv_tts > 0.0, f"|z_V| 之和 = {zv_tts:.2f}")
        else:
            check("对照：不丢时 TTS 行 z_T 非零", zt_tts > 0.0,
                  f"|z_T| 之和 = {zt_tts:.2f}")
            check("对照：不丢时 S2T 行 z_V 非零", zv_s2t > 0.0,
                  f"|z_V| 之和 = {zv_s2t:.2f}")

    print()
    print(f"  结论：{len(FAIL)} 项未通过" + (f" -> {FAIL}" if FAIL else "（全部通过）"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
