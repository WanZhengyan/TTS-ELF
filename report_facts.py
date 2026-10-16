#!/usr/bin/env python
"""Measure the numbers the report states about the two-source run.

Nothing here is copied by hand: parameter counts come from the constructed
model, the task mix from real batches drawn by the trainer's own sampler and
collate, and the data figures from the cache metas. Writes out/dual/facts.json.
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_unified as T
from unified import UnifiedELF, TTS, S2T


def main():
    R = lambda c: os.path.join(HERE, c)
    tts_c = R("cache_lj13k")
    s2t_cs = [R("cache_libri_tc100_lv3"), R("cache_libri_train_clean_360"),
              R("cache_libri_train_other_500")]
    ev_c = R("cache_libri_test_clean_lv3")
    F = {}

    # ---- parameters, from the model the run builds
    tb = T.LibriCache(tts_c, "all")
    meta = tb.meta
    m = UnifiedELF("ELF-B", char_vocab=meta["char_vocab_size"], max_length=1648,
                   whisper_dim=1280, use_cond=True, spk_dim=0)
    groups = {"whisper_proj": 0, "self_cond_proj": 0, "voice_out": 0,
              "char_embed+cond_norm": 0, "backbone": 0}
    for n, p in m.named_parameters():
        k = ("whisper_proj" if n.startswith("whisper_proj") else
             "self_cond_proj" if "self_cond_proj" in n else
             "voice_out" if n.startswith("voice_out") else
             "char_embed+cond_norm" if (n.startswith("char_embed") or n.startswith("cond_norm"))
             else "backbone")
        groups[k] += p.numel()
    F["params"] = groups
    F["params_total"] = sum(groups.values())
    F["input_width"] = m.input_width
    from transformers import WhisperConfig, WhisperModel
    cfgw = WhisperConfig.from_pretrained("openai/whisper-large-v3")
    with torch.device("meta"):
        enc = WhisperModel(cfgw).encoder
    F["whisper_encoder_params"] = sum(p.numel() for p in enc.parameters())

    # ---- data
    def stats(b):
        recs = b.recs
        h = sum(r["T"] for r in recs) / b.meta["frame_rate"] / 3600
        return {"clips": len(recs), "hours": h, "speakers": b.meta["n_speakers"],
                "t5_mean": float(np.mean([len(r["t5"]) for r in recs])),
                "frames_per_t5": float(np.mean([r["T"] / max(len(r["t5"]), 1) for r in recs])),
                "frames_per_char": float(np.mean([r["T"] / max(len(r["chars"]), 1) for r in recs])),
                "sec_mean": float(np.mean([r["T"] for r in recs]) / b.meta["frame_rate"]),
                "mel_mean": b.meta["latent_mean"], "mel_std": b.meta["latent_std"]}
    F["tts"] = stats(tb)
    sbs = [T.LibriCache(c, "all") for c in s2t_cs]
    F["s2t"] = {os.path.basename(c): stats(b) for c, b in zip(s2t_cs, sbs)}
    wgb = 0.0
    for b in sbs:
        wgb += b.meta["whisper_total"] * b.meta["whisper_dim"] * 2 / 1e9
    F["whisper_gb"] = wgb
    F["whisper_model"] = sbs[0].meta["whisper_model"]
    eb = T.LibriCache(ev_c, "all")
    F["test_clean_whole"] = len(eb)

    # ---- the task mix in real batches: the trainer's sampler and collate
    ref = (meta["latent_mean"], meta["latent_std"])
    tds = T.Tagged(tb, TTS, ref)
    sds = [T.Tagged(b, S2T, ref) for b in sbs]
    ds = torch.utils.data.ConcatDataset([tds] + sds)
    n_tts, n_s2t = len(tds), sum(len(d) for d in sds)
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("t5-small")
    itos = {v: k for k, v in meta["stoi"].items()}
    coll = T.make_collate(1648, 0.5, (0.0, 0.3), max_text=112, itos=itos, tk=tk,
                          layout="hybrid", whisper_rate=50.0,
                          frame_rate=meta["frame_rate"], whisper_dim=1280)
    it = iter(T.TaskSampler(n_tts, n_s2t, 0.5, seed=[0, 0, 17]))
    rows = {"tts": 0, "s2t": 0}
    pos = {"tts": 0.0, "s2t": 0.0}
    pfrac, tsec = [], []
    for _ in range(40):
        b = coll([ds[next(it)] for _ in range(32)])
        is_s2t = (b["task"] == S2T)
        tgt_v = (b["v_act"] * (1 - b["prompt_v"])).sum(1)
        tgt_t = (b["t_act"] * (1 - b["prompt_t"])).sum(1)
        rows["s2t"] += int(is_s2t.sum()); rows["tts"] += int((~is_s2t).sum())
        pos["s2t"] += float(tgt_t[is_s2t].sum()); pos["tts"] += float(tgt_v[~is_s2t].sum())
        va = b["v_act"].sum(1)
        for i in torch.nonzero(~is_s2t).flatten().tolist():
            pfrac.append(float(b["prompt_v"][i].sum() / va[i]))
            tsec.append(float(va[i]) / meta["frame_rate"])
    tot_r, tot_p = sum(rows.values()), sum(pos.values())
    F["mix"] = {"batches": 40, "rows": rows,
                "row_share_tts": rows["tts"] / tot_r,
                "pos_share_tts": pos["tts"] / tot_p,
                "tts_target_pos_per_row": pos["tts"] / max(rows["tts"], 1),
                "s2t_target_pos_per_row": pos["s2t"] / max(rows["s2t"], 1),
                "tts_prompt_frac_mean": float(np.mean(pfrac)),
                "tts_row_sec_mean": float(np.mean(tsec)),
                "s2t_share_if_union_shuffled": n_s2t / (n_tts + n_s2t)}
    os.makedirs(R("out/dual"), exist_ok=True)
    json.dump(F, open(R("out/dual/facts.json"), "w"), indent=1)
    print(json.dumps(F, indent=1))


if __name__ == "__main__":
    main()
