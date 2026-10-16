#!/usr/bin/env python
"""Check the invariants the headline numbers depend on, on real batches.

Written after two silent failures that each produced a plausible-looking
number: an evaluation that scored 0 clips and printed zeros, and a SIM-o of
0.96 that was measuring the ground-truth prompt copied into the output. Both
would have been caught by asserting what the batch is supposed to contain, so
that is what this does. Every check prints the measured quantity, not just
PASS, because a check that only prints PASS hides the margin it passed by.

It also tests, rather than asserts, the explanation offered for why TTS has
twice the exposure gap of S2T: that the uniform text upsampling
t5_idx[j] = floor(j*N/T) gives the model a WRONG alignment. Whisper word
timestamps on the real recording give a reference alignment to measure against.
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from unified import TTS, S2T
from train_unified import LibriCache, make_collate

FAIL = []


def check(name, ok, detail):
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}")
    print(f"         {detail}")
    if not ok:
        FAIL.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(HERE, "cache_libri_test_clean"))
    ap.add_argument("--libri_root", default=os.path.join(HERE, "data", "LibriSpeech"))
    ap.add_argument("--libri_split", default="test-clean")
    ap.add_argument("--max_length", type=int, default=1648)
    ap.add_argument("--max_text", type=int, default=112)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--prompt_sec", type=float, default=3.0)
    ap.add_argument("--align_n", type=int, default=40)
    args = ap.parse_args()

    ds = LibriCache(args.cache, "all")
    meta = ds.meta
    fr = meta["frame_rate"]
    itos = {v: k for k, v in meta["stoi"].items()}
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("t5-small")
    tok0 = args.max_length - args.max_text

    print("=" * 70)
    print("1. 混合批次的不变量（p_s2t=0.5，两个任务在同一批）")
    print("=" * 70)
    coll = make_collate(args.max_length, 0.5, (0.0, 0.3), rng_seed=7,
                        max_text=args.max_text, itos=itos, tk=tk,
                        layout="hybrid",
                        whisper_rate=meta.get("whisper_rate", 0.0),
                        frame_rate=fr)
    b = next(iter(DataLoader(ds, batch_size=args.batch, collate_fn=coll)))
    task = b["task"].numpy()
    tts, s2t = task == TTS, task == S2T
    ta, va = b["t_act"].numpy(), b["v_act"].numpy()
    pt, pv = b["prompt_t"].numpy(), b["prompt_v"].numpy()
    idx = b["t5_idx"].numpy()
    wm = b["w_mask"].numpy()
    ch = b["chars"].numpy()

    check("TTS 行：所有文本位置都是干净条件 (prompt_t == t_act)",
          np.array_equal(pt[tts], ta[tts]),
          f"不相等的位置数 {int((pt[tts] != ta[tts]).sum())}")
    check("S2T 行：文本全部是待生成目标 (prompt_t 全 0)",
          pt[s2t].sum() == 0,
          f"prompt_t 非零位置数 {int(pt[s2t].sum())}")
    check("S2T 行：mel 全部是干净条件 (prompt_v == v_act)",
          np.array_equal(pv[s2t], va[s2t]),
          f"不相等的位置数 {int((pv[s2t] != va[s2t]).sum())}")
    check("文本 latent 与 Whisper 特征永不同时非零",
          ((idx != -1) * wm).sum() == 0,
          f"重叠位置数 {int(((idx != -1) * wm).sum())}")
    check("S2T 行：cond(chars) 被任务门控为 0（泄漏修复）",
          (ch[s2t] != 0).sum() == 0,
          f"非零字符位置数 {int((ch[s2t] != 0).sum())}")
    check("Whisper 只写在 mel 区，不碰 token 区",
          wm[:, tok0:].sum() == 0,
          f"token 区的 Whisper 位置数 {int(wm[:, tok0:].sum())}")
    check("token 区整块都是文本活跃（不泄漏答案长度）",
          np.all(ta[:, tok0:tok0 + args.max_text] == 1.0),
          f"token 区 t_act 均值 {ta[:, tok0:].mean():.4f}，应为 1.0")

    print()
    print("=" * 70)
    print("2. prompt 的绝对秒数（文献协议）")
    print("=" * 70)
    coll_p = make_collate(args.max_length, 0.0, (0.0, 0.0), rng_seed=7,
                          max_text=args.max_text, itos=itos, tk=tk,
                          layout="hybrid",
                          whisper_rate=meta.get("whisper_rate", 0.0),
                          frame_rate=fr, prompt_sec=args.prompt_sec)
    ds4 = LibriCache(args.cache, "all", dur_range=(4.0, 10.0))
    bp = next(iter(DataLoader(ds4, batch_size=args.batch, collate_fn=coll_p)))
    K = bp["n_prompt"].numpy()
    T = bp["v_act"].numpy().sum(1)
    secs = K / fr
    check(f"每条的 prompt 都是 {args.prompt_sec}s",
          np.allclose(secs, args.prompt_sec, atol=1.0 / fr),
          f"实测 {secs.min():.3f}-{secs.max():.3f}s（±1 帧 = {1/fr:.4f}s）")
    check("prompt 严格短于音频（否则没有要生成的部分）",
          np.all(K < T),
          f"prompt 占比 {100*(K/T).min():.0f}%-{100*(K/T).max():.0f}%")
    print(f"         提示：4s 的 clip 上 3s prompt = 75%，"
          f"而训练只到 prompt_hi=0.3")

    print()
    print("=" * 70)
    print("3. 生成的 mel 在 prompt 区是真实 mel 吗（SIM-o 0.96 的来源）")
    print("=" * 70)
    mel = bp["mel"].numpy()
    pvp = bp["prompt_v"].numpy()
    # prompt_v marks where unified.restore() pins z_v to the clean value, so
    # anything scored over those frames is scoring the recording, not the model
    nz = int(pvp.sum())
    frac = float((pvp.sum(1) / np.maximum(T, 1)).mean())
    check("prompt 区会被 restore() 钉死为真实 mel —— 打分必须排除",
          nz > 0,
          f"每条平均 {pvp.sum(1).mean():.0f} 帧被钉死，占音频 {100*frac:.1f}%"
          f" -> 整句打分里有这么多是真音频")

    print()
    print("=" * 70)
    print("4. 均匀拉伸 vs 真实对齐（TTS 条件信号的误差）")
    print("=" * 70)
    import glob
    id2text = {}
    for tr in glob.glob(os.path.join(args.libri_root, args.libri_split,
                                     "*", "*", "*.trans.txt")):
        with open(tr, encoding="utf-8") as f:
            for line in f:
                uid, _, txt = line.strip().partition(" ")
                id2text[uid] = txt
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        print("  跳过：没有 GPU，无法跑 Whisper 取词级时间戳")
    else:
        import torchaudio
        from transformers import pipeline
        asr = pipeline("automatic-speech-recognition",
                       model="openai/whisper-large-v3", device=0,
                       torch_dtype=torch.float16)
        errs, rels = [], []
        done = 0
        for r in ds4.recs:
            if done >= args.align_n:
                break
            uid = r["id"]
            if uid not in id2text:
                continue
            spk, chap, _ = uid.split("-")
            p = os.path.join(args.libri_root, args.libri_split, spk, chap,
                             uid + ".flac")
            if not os.path.exists(p):
                continue
            w, sr = torchaudio.load(p)
            w = (w.mean(0) if w.shape[0] > 1 else w[0])
            w16 = torchaudio.functional.resample(w, sr, 16000).numpy()
            out = asr(w16.copy(), return_timestamps="word",
                      generate_kwargs={"language": "en"})
            chunks = [c for c in out.get("chunks", [])
                      if c.get("timestamp") and c["timestamp"][0] is not None]
            if len(chunks) < 3:
                continue
            dur = r["dur"]
            nw = len(chunks)
            # the uniform assumption our t5_idx encodes: unit k starts at
            # k/nw of the clip. Compare against where the word really starts.
            for k, c in enumerate(chunks):
                uni = dur * k / nw
                real = float(c["timestamp"][0])
                errs.append(abs(uni - real))
            rels.append(dur / nw)
            done += 1
        if errs:
            e = np.array(errs)
            print(f"  在 {done} 条、{len(errs)} 个词上测量：")
            print(f"    均匀假设的起始时刻误差  均值 {e.mean()*1000:.0f} ms"
                  f"  中位 {np.median(e)*1000:.0f} ms  p90 {np.percentile(e,90)*1000:.0f} ms")
            print(f"    平均每词时长 {np.mean(rels)*1000:.0f} ms")
            print(f"    -> 误差是一个词时长的 {e.mean()/np.mean(rels):.2f} 倍")
            print(f"    -> 折算成 mel 帧：平均偏 {e.mean()*fr:.0f} 帧")
            check("均匀拉伸的对齐误差小于半个词（否则条件信号在误导模型）",
                  e.mean() < 0.5 * np.mean(rels),
                  f"误差/词长 = {e.mean()/np.mean(rels):.2f}，阈值 0.50")

    print()
    print("=" * 70)
    print(f"结论：{len(FAIL)} 项未通过" + (f" -> {FAIL}" if FAIL else "（全部通过）"))
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
