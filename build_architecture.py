#!/usr/bin/env python
"""Derive architecture.html from report.html so the two cannot disagree.

architecture.html used to be hand-written and drifted from the report: it kept
a speaker-embedding channel after we dropped it, and its parameter counts
described a model that was never trained. Extracting the architecture, input
and example sections from the generated report makes them identical by
construction.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def section(s, sid):
    a = s.index(f'<section id="{sid}">')
    b = s.index("</section>", a) + len("</section>")
    return s[a:b]


def main():
    s = open(os.path.join(HERE, "report.html")).read()
    style = re.search(r"<style>.*?</style>", s, flags=re.S).group(0)
    fonts = re.search(r'<link rel="stylesheet"[^>]*>', s).group(0)
    parts = [section(s, sid) for sid in ("arch", "inputs", "ex")]
    # renumber the eyebrows 1/2/8 -> 1/2/3 for this shorter page
    parts[2] = parts[2].replace('<p class="eyebrow">8</p>', '<p class="eyebrow">3</p>', 1)
    out = "\n".join([
        "<title>统一模型的完整架构</title>", fonts, style, '<div class="wrap">',
        '<p class="eyebrow">统一 TTS + S2T · 2026-10-15</p>',
        "<h1>统一模型的完整架构</h1>",
        '<p class="lede prose">一个 ELF-B 主干，两个任务，零任务专属层。这页画清每个模块、'
        "每处维度变换和每个输入通道的来源，最后是两个方向的真实例子。"
        "数字都从 ckpt 和 cache 的 meta 里读出来，和实验报告完全一致。</p>",
        '<hr class="r">', parts[0], '<hr class="r">', parts[1],
        '<hr class="r">', parts[2], "</div>"])
    p = os.path.join(HERE, "architecture.html")
    open(p, "w").write(out)
    vis = re.sub(r"<style>.*?</style>", "", out, flags=re.S)
    vis = re.sub(r"base64,[A-Za-z0-9+/=]+", "", vis)
    left = re.findall(r"x-vector|xvec|spk_proj|UniSpeechSat|说话人条件", vis)
    print(f"wrote {p}  ({os.path.getsize(p)/2**20:.2f} MB)")
    print(f"  section {out.count('<section')}/{out.count('</section>')}  "
          f"div {out.count('<div')}/{out.count('</div>')}  "
          f"svg {out.count('<svg')}/{out.count('</svg>')}  audio {out.count('<audio')}")
    print(f"  可见文字里的 x-vector 相关词: {len(left)}")
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main())
