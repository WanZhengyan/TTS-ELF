#!/usr/bin/env python
"""The LJSpeech test list, built the way F5-TTS describes its own.

F5-TTS small's LJSpeech column (arXiv 2410.06885, Table 9) is measured on
"in-set tests (1000 samples organized with 4 to 7 seconds to infer with an
around 3-second prompt)". The list itself is not in their repo (only the
LibriSpeech-PC one is), so this rebuilds one under the same rules, with a
fixed seed:

  target  1000 clips of 4.0-7.0 s, drawn without replacement
  prompt  a DIFFERENT clip of 2.5-3.5 s ("around 3 s"), drawn with
          replacement (only 881 such clips exist); the two ranges are disjoint
          so a prompt can never be its own target

In-set means these clips are in the training data, as in F5: a single-speaker
corpus has no unseen speaker to hold out. Written in F5's .lst layout:
prompt_uid, prompt_dur, prompt_text, target_uid, target_dur, target_text.
"""
import argparse
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(HERE, "cache_lj13k"))
    ap.add_argument("--lj_root", default=os.path.join(HERE, "data", "LJSpeech-1.1"))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "lj_eval",
                                                  "lj_inset_1000.lst"))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    recs = pickle.load(open(os.path.join(a.cache, "index.pt"), "rb"))["records"]
    text = {}
    with open(os.path.join(a.lj_root, "metadata.csv"), encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("|")
            text[p[0]] = p[2].strip()          # normalized transcript, as F5 trains on
    tgt = sorted([r for r in recs if 4.0 <= r["dur"] <= 7.0], key=lambda r: r["id"])
    pro = sorted([r for r in recs if 2.5 <= r["dur"] <= 3.5], key=lambda r: r["id"])
    rng = np.random.default_rng(a.seed)
    ti = rng.choice(len(tgt), a.n, replace=False)
    pi = rng.choice(len(pro), a.n, replace=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for i, j in zip(ti, pi):
            t, p = tgt[i], pro[j]
            f.write(f"{p['id']}\t{p['dur']:.3f}\t{text[p['id']]}\t"
                    f"{t['id']}\t{t['dur']:.3f}\t{text[t['id']]}\n")
    td = np.array([tgt[i]["dur"] for i in ti]); pd = np.array([pro[j]["dur"] for j in pi])
    print(f"wrote {a.out}: {a.n} pairs from {len(tgt)} targets (4-7 s) and "
          f"{len(pro)} prompts (2.5-3.5 s)")
    print(f"  target dur mean {td.mean():.2f} s, prompt dur mean {pd.mean():.2f} s, "
          f"{len(set(pi))} distinct prompts")


if __name__ == "__main__":
    main()
