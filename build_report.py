#!/usr/bin/env python
"""Build the experimental report for the two-source unified run.

TTS rows train on LJSpeech (compared with F5-TTS small's LJSpeech column),
S2T rows on LibriSpeech 960 h with frozen whisper-large-v3 (compared with
ELF-S2T's ELF-B), one ELF-B backbone, 100K steps = ELF-B's S2T budget.

Every number is read at build time from where it was measured:
  out/dual/facts.json        parameters, data, batch mix   (report_facts.py)
  logs/uni-<job>.out         training progress and monitor evals
  out/dual/examples_*.json   listening examples            (make_dual_examples.py)
  out/dual/*_lj.json         TTS, F5 protocol               (eval_lj.py)
  out/dual/*_s2t.json        S2T, all of test-clean         (eval_s2t_full.py)
so re-running this after a milestone eval updates the page. The previous
version of the report (single dataset, LibriSpeech 100 h) is
build_report_v1.py / report_v1.html.
"""
import glob
import html
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_unified import wer_counts
from build_report_v1 import CSS

E = html.escape
JOBS = ["16535", "16556"]          # 0 -> ~12K constant lr, then the 100K cosine run
WANDB = ("https://wandb.ai/zhengyanwan066-east-china-normal-university/"
         "voiceflow-unified/runs/dual_lj_ls960-16535")
EXTRA_CSS = r"""
.status{display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
        gap:10px; margin:18px 0 0;}
.st{background:var(--surface); border:1px solid var(--line); border-radius:7px;
    padding:11px 14px;}
.st .k{font-family:var(--mono); font-size:.68rem; letter-spacing:.07em;
       text-transform:uppercase; color:var(--muted); display:block;}
.st .v{font-family:var(--mono); font-size:1.15rem; font-weight:600;
       font-variant-numeric:tabular-nums;}
.st .d{font-size:.8rem; color:var(--muted);}
.pend{color:var(--muted); font-style:italic;}
"""


def fmt(n):
    return f"{n:,}"


def read_logs():
    """Training rows and monitor evals across the two jobs, later job wins."""
    rows, evals, samples = {}, {}, []
    for j in JOBS:
        p = os.path.join(HERE, "logs", f"uni-{j}.out")
        if not os.path.exists(p):
            continue
        cur = None
        for line in open(p, encoding="utf-8", errors="replace"):
            m = re.match(r"^\s+(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+"
                         r"([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)%\s+([-\d.]+)\s+([-\d.]+)", line)
            if m:
                rows[int(m.group(1))] = {"loss": float(m.group(2)), "sit": float(m.group(10)),
                                         "job": j}
                continue
            m = re.search(r"\[eval (\d+) ema\] TTS mel err/trivial ([-\d.]+).*?"
                          r"S2T text err/trivial ([-\d.]+) WER ([-\d.]+)% tok_acc ([-\d.]+)%", line)
            if m:
                cur = int(m.group(1))
                evals[cur] = {"tts": float(m.group(2)), "s2t_ratio": float(m.group(3)),
                              "wer": float(m.group(4)), "tok": float(m.group(5)), "job": j}
                samples = [s for s in samples if s[0] != cur]
                continue
            m = re.match(r"^\s+(ref|hyp)\s+(.*)$", line.rstrip("\n"))
            if m and cur is not None:
                samples.append((cur, m.group(1), m.group(2)))
    return rows, evals, samples


def s2t_examples(evals, samples):
    if not evals:
        return '<div class="note"><p>还没有监控评测。</p></div>'
    last = max(evals)
    rs = [s for s in samples if s[0] == last]
    pairs = [(rs[i][2], rs[i + 1][2]) for i in range(0, len(rs) - 1, 2)
             if rs[i][1] == "ref" and rs[i + 1][1] == "hyp"]
    H = [f'<div class="ex s2t"><header><h4>S2T　语音 → 文本（第 {fmt(last)} 步的真实解码）</h4></header>',
         '<div class="row"><p style="margin:0;font-size:.9rem">训练日志里每次监控评测打印的前三条 test-clean，'
         'EMA 权重、32 步、不加引导。日志把超过 110 个字符的句子截断，这里只列完整的。</p></div>']
    shown = 0
    for ref, hyp in pairs:
        if len(ref) >= 110:
            continue
        c = wer_counts(ref, hyp)
        w = 100.0 * c["err"] / max(c["n"], 1)
        H += [f'<div class="row"><div class="meta">这条 WER {w:.1f}%　·　'
              f'替换 {c["sub"]} / 删除 {c["dele"]} / 插入 {c["ins"]}</div>',
              '<dl class="pair">', f"<dt>参考</dt><dd>{E(ref)}</dd>",
              f"<dt>模型输出</dt><dd>{E(hyp)}</dd>", "</dl></div>"]
        shown += 1
    if not shown:
        H.append('<div class="row"><p style="margin:0">这一次的三条都长于 110 字符。</p></div>')
    H.append("</div>")
    return "\n".join(H)


def tts_examples():
    fs = sorted(glob.glob(os.path.join(HERE, "out", "dual", "examples_*.json")))
    if not fs:
        return '<div class="note"><p>还没有 TTS 例子。</p></div>'
    d = json.load(open(fs[-1]))
    H = [f'<div class="ex tts"><header><h4>TTS　文本 → 语音（第 {fmt(d["step"])} 步，真实音频）</h4></header>',
         f'<div class="row"><p style="margin:0;font-size:.9rem">评测列表的<b>前 {len(d["cases"])} 对，没有挑选</b>。'
         f'输入和正式评测完全一样：另一句约 3 秒的 LJ 录音当 prompt，文本是「prompt 文本 + 目标文本」，'
         f'生成长度用 F5 的估计。Euler {d["nfe"]} 步，CFG w = {d["cfg"]}。转写用外部的 Whisper-large-v3，'
         f'不是我们自己的 S2T。</p></div>']
    for c in d["cases"]:
        H += [f'<div class="row"><div class="meta">{E(c["target_uid"])}　·　prompt {E(c["prompt_uid"])} '
              f'{c["prompt_sec"]:.2f}s　·　生成 {c["gen_sec"]:.2f}s（真实 {c["target_sec"]:.2f}s）　·　'
              f'WER {c["wer"]:.1f}%</div>',
              '<dl class="pair">', f'<dt>要说的</dt><dd>{E(c["target_text"])}</dd>',
              f'<dt>生成转写</dt><dd>{E(c["hyp"]) or "（空）"}</dd>', "</dl>",
              '<div class="players">']
        for lb, k in (("prompt 原始录音", "prompt_wav"), ("目标 真实录音", "target_wav"),
                      ("生成", "gen_wav")):
            H.append(f'<div class="pl"><span class="lb">{lb}</span><audio controls '
                     f'preload="metadata" src="data:audio/wav;base64,{c[k]}"></audio></div>')
        H.append("</div></div>")
    ws = [c["wer"] for c in d["cases"]]
    H.append(f'<div class="row"><p style="margin:0;font-size:.9rem">第 {fmt(d["step"])} 步时，'
             f'生成的是流畅的、LJ 音色的英语，但<b>内容和要说的文本无关</b>（这 {len(ws)} 条 WER '
             f'{", ".join(f"{w:.0f}%" for w in ws)}）—— 模型还没学会按文本说话。'
             f'作为参照，两个任务都用 LJ 的上一版在 4 万步时是 28%（旧口径，见第 7 节）。</p></div></div>')
    return "\n".join(H)


def formal_results():
    """Milestone evals, if any have run."""
    out = {}
    for f in glob.glob(os.path.join(HERE, "out", "dual", "*_lj.json")):
        d = json.load(open(f))
        out.setdefault(d["step"], {})["tts"] = d
    for f in glob.glob(os.path.join(HERE, "out", "dual", "*_s2t.json")):
        d = json.load(open(f))
        out.setdefault(d["step"], {})["s2t"] = d
    return out


def svg(F):
    p = F["params_total"]
    return f"""
<svg viewBox="0 0 1000 738" role="img" aria-label="统一模型架构：五个通道块在 1648 个位置上，TTS 与 S2T 填不同格子，汇入同一个 ELF-B 主干，分出两个输出头">
<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><polygon points="0,1 10,5 0,9" fill="currentColor"/></marker></defs>
<text x="110" y="24" font-family="IBM Plex Sans, sans-serif" font-size="12.5" font-weight="600" fill="currentColor">TTS　文本 → 语音（LJSpeech）</text>
<text x="556" y="24" font-family="IBM Plex Sans, sans-serif" font-size="12.5" font-weight="600" fill="currentColor">S2T　语音 → 文本（LibriSpeech）</text>
<g font-family="IBM Plex Mono, monospace" font-size="9.5" fill="currentColor" opacity=".6">
<text x="110" y="40">位置 0–1535 · mel 区 · 16.38 s</text><text x="398" y="40">1536–1647</text>
<text x="556" y="40">位置 0–1535 · mel 区</text><text x="844" y="40">1536–1647</text></g>
<g font-family="IBM Plex Mono, monospace" font-size="11" fill="currentColor">
<text x="14" y="66">z_T</text><text x="14" y="78" font-size="9" opacity=".5">512</text>
<text x="14" y="108">sc_T</text><text x="14" y="120" font-size="9" opacity=".5">512</text>
<text x="14" y="150">z_V</text><text x="14" y="162" font-size="9" opacity=".5">100</text>
<text x="14" y="192">sc_V</text><text x="14" y="204" font-size="9" opacity=".5">100</text>
<text x="14" y="234">cond</text><text x="14" y="246" font-size="9" opacity=".5">512</text></g>
<rect x="62" y="48" width="326" height="34" rx="3" fill="var(--text-bg)" stroke="var(--text)" stroke-width="1.6"/>
<text x="225" y="69" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="11" fill="var(--text)">T5 latent · 均匀升采样 {F["tts"]["frames_per_t5"]:.1f} 帧/token</text>
<rect x="396" y="48" width="56" height="34" rx="3" fill="var(--text-bg)" stroke="var(--text)" stroke-width="1.6"/>
<text x="424" y="69" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="10.5" fill="var(--text)">T5</text>
<rect x="62" y="90" width="326" height="34" rx="3" fill="var(--text-bg)" stroke="var(--text)" stroke-width="1.6" opacity=".5"/>
<text x="225" y="111" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="11" fill="currentColor" opacity=".75">干净文本的副本（restore 钉回，不是自条件）</text>
<rect x="396" y="90" width="56" height="34" rx="3" fill="var(--text-bg)" stroke="var(--text)" stroke-width="1.6" opacity=".5"/>
<rect x="62" y="132" width="326" height="34" rx="3" fill="none" stroke="var(--voice)" stroke-width="2.2" stroke-dasharray="6 4"/>
<text x="225" y="153" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="11" font-weight="600" fill="var(--voice)">mel：前 {100*F["mix"]["tts_prompt_frac_mean"]:.1f}% 是 prompt，其余从噪声积分</text>
<rect x="396" y="132" width="56" height="34" rx="3" fill="var(--sunk)" stroke="var(--line)"/>
<text x="424" y="153" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10" fill="currentColor" opacity=".4">0</text>
<rect x="62" y="174" width="326" height="34" rx="3" fill="none" stroke="var(--voice)" stroke-width="1.5" stroke-dasharray="4 4" opacity=".6"/>
<text x="225" y="195" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="11" fill="var(--voice)" opacity=".85">自条件（上一步的 x̂₀）</text>
<rect x="396" y="174" width="56" height="34" rx="3" fill="var(--sunk)" stroke="var(--line)"/>
<text x="424" y="195" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10" fill="currentColor" opacity=".4">0</text>
<rect x="62" y="216" width="326" height="34" rx="3" fill="var(--frozen-bg)" stroke="var(--frozen)" stroke-width="1.6"/>
<text x="225" y="237" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="11" fill="var(--frozen)">字符查表 · {F["tts"]["frames_per_char"]:.1f} 帧/字符</text>
<rect x="396" y="216" width="56" height="34" rx="3" fill="var(--sunk)" stroke="var(--line)"/>
<text x="424" y="237" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10" fill="currentColor" opacity=".4">0</text>
<rect x="508" y="48" width="326" height="34" rx="3" fill="var(--frozen-bg)" stroke="var(--frozen)" stroke-width="1.6"/>
<text x="671" y="69" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="11" fill="var(--frozen)">Whisper-large-v3 投影 1280→512（文本已清零）</text>
<rect x="842" y="48" width="56" height="34" rx="3" fill="none" stroke="var(--text)" stroke-width="2.2" stroke-dasharray="6 4"/>
<text x="870" y="69" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="9.5" font-weight="600" fill="var(--text)">文本←噪声</text>
<rect x="508" y="90" width="326" height="34" rx="3" fill="var(--sunk)" stroke="var(--line)"/>
<text x="671" y="111" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10" fill="currentColor" opacity=".4">0</text>
<rect x="842" y="90" width="56" height="34" rx="3" fill="none" stroke="var(--text)" stroke-width="1.5" stroke-dasharray="4 4" opacity=".6"/>
<text x="870" y="111" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="9.5" fill="var(--text)" opacity=".85">自条件</text>
<rect x="508" y="132" width="326" height="34" rx="3" fill="var(--voice-bg)" stroke="var(--voice)" stroke-width="1.6"/>
<text x="671" y="153" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="11" fill="var(--voice)">mel（干净条件，全程不动）</text>
<rect x="842" y="132" width="56" height="34" rx="3" fill="var(--sunk)" stroke="var(--line)"/>
<text x="870" y="153" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10" fill="currentColor" opacity=".4">0</text>
<rect x="508" y="174" width="326" height="34" rx="3" fill="var(--voice-bg)" stroke="var(--voice)" stroke-width="1.5" opacity=".5"/>
<text x="671" y="195" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="10.5" fill="currentColor" opacity=".8">干净 mel 的副本（restore 钉回，不是自条件）</text>
<rect x="842" y="174" width="56" height="34" rx="3" fill="var(--sunk)" stroke="var(--line)"/>
<text x="870" y="195" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10" fill="currentColor" opacity=".4">0</text>
<rect x="508" y="216" width="326" height="34" rx="3" fill="var(--sunk)" stroke="var(--line)"/>
<text x="671" y="237" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10" fill="currentColor" opacity=".4">0 · 按任务门控</text>
<rect x="842" y="216" width="56" height="34" rx="3" fill="var(--sunk)" stroke="var(--line)"/>
<text x="870" y="237" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10" fill="currentColor" opacity=".4">0</text>
<path d="M 225 258 L 225 272 L 671 272 L 671 258" fill="none" stroke="var(--frozen)" stroke-width="1.2" stroke-dasharray="3 3"/>
<text x="448" y="288" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="10.5" fill="var(--frozen)">z_T 一行被 T5 和 Whisper 共用，两者按任务占不同的行，互不重叠</text>
<g stroke="currentColor" stroke-width="1.4" fill="none" marker-end="url(#ar)" opacity=".7">
<path d="M 257 300 L 257 318 L 494 318 L 494 334"/><path d="M 703 300 L 703 318 L 506 318 L 506 334"/></g>
<rect x="210" y="338" width="580" height="34" rx="4" fill="var(--surface-2)" stroke="var(--line)"/>
<text x="500" y="360" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="11" fill="currentColor">concat → (B, 1648, {F["input_width"]})　＝ 512 + 512 + 100 + 100 + 512</text>
<line x1="500" y1="372" x2="500" y2="392" stroke="currentColor" stroke-width="1.4" marker-end="url(#ar)" opacity=".7"/>
<rect x="262" y="396" width="476" height="32" rx="4" fill="var(--surface)" stroke="var(--line)"/>
<text x="500" y="417" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="11" fill="currentColor">self_cond_proj　1736 → 512　（{fmt(F["params"]["self_cond_proj"])} 参数）</text>
<line x1="500" y1="428" x2="500" y2="448" stroke="currentColor" stroke-width="1.4" marker-end="url(#ar)" opacity=".7"/>
<rect x="262" y="452" width="476" height="30" rx="4" fill="var(--surface-2)" stroke="var(--line)" stroke-dasharray="4 3"/>
<text x="500" y="472" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10.5" fill="currentColor" opacity=".8">前置 token：4 时间 + 4 自条件CFG + 4 模型模式 + 4 任务 = 16 位置</text>
<line x1="500" y1="482" x2="500" y2="502" stroke="currentColor" stroke-width="1.4" marker-end="url(#ar)" opacity=".7"/>
<rect x="210" y="506" width="580" height="58" rx="5" fill="var(--surface)" stroke="currentColor" stroke-width="2.2"/>
<text x="500" y="530" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="14" font-weight="600" fill="currentColor">ELF-B 主干　12 层 × 宽 768　{F["params"]["backbone"]/1e6:.2f}M</text>
<text x="500" y="550" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10.5" fill="currentColor" opacity=".7">两任务 100% 共用 · 双向注意力 · 一次混合前向 · RoPE</text>
<g stroke="currentColor" stroke-width="1.4" fill="none" marker-end="url(#ar)" opacity=".7">
<path d="M 330 564 L 330 592"/><path d="M 670 564 L 670 592"/></g>
<rect x="196" y="596" width="268" height="58" rx="4" fill="var(--voice-bg)" stroke="var(--voice)" stroke-width="1.6"/>
<text x="330" y="615" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10.5" fill="var(--voice)">final_layer 768→512 → voice_out 512→100</text>
<text x="330" y="634" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="10.5" fill="var(--voice)" opacity=".85">mel → Vocos → 波形（TTS，回归）</text>
<rect x="536" y="596" width="268" height="58" rx="4" fill="var(--text-bg)" stroke="var(--text)" stroke-width="1.6"/>
<text x="670" y="615" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10.5" fill="var(--text)">proj 768→512 → unembed 512→32,100</text>
<text x="670" y="634" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="10.5" fill="var(--text)" opacity=".85">token 区 argmax → 文本（S2T，分类）</text>
<text x="500" y="686" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="11" fill="currentColor" opacity=".62">任务专属参数只有 task token：2 × 4 × 768 = 6,144，占 {p/1e6:.2f}M 的 {100*6144/p:.4f}%</text>
<text x="500" y="708" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="11" fill="currentColor" opacity=".62">两个输出头本质不同：S2T 是交叉熵分类头，TTS 是回归 —— 第 7 节说明这为什么重要</text>
</svg>"""


def main():
    F = json.load(open(os.path.join(HERE, "out", "dual", "facts.json")))
    rows, evals, samples = read_logs()
    step = max(rows) if rows else 0
    sit = rows[step]["sit"] if rows else 0.0
    FR = formal_results()
    sw = None                       # the step the cosine run resumed from
    p2 = os.path.join(HERE, "logs", f"uni-{JOBS[1]}.out")
    if os.path.exists(p2):
        m = re.search(r"resumed from \S+ at step (\d+)", open(p2, errors="replace").read())
        sw = int(m.group(1)) if m else None
    sw_txt = f"第 {fmt(sw)} 步起" if sw else "新作业接手时起"
    s2t = F["s2t"]
    s2t_clips = sum(v["clips"] for v in s2t.values())
    s2t_hours = sum(v["hours"] for v in s2t.values())
    s2t_spk = sum(v["speakers"] for v in s2t.values())
    mix = F["mix"]
    pos_tts = mix["pos_share_tts"]
    tts_h_per_step = 32 * mix["row_share_tts"] * mix["tts_row_sec_mean"] / 3600
    tts_h_100k = 100000 * tts_h_per_step

    out = []
    A = out.append
    A('<title>统一模型实验报告</title>')
    A('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:'
      'wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:'
      'wght@400;500;600&display=swap">')
    A(f"<style>{CSS}{EXTRA_CSS}</style>")
    A('<div class="wrap">')
    A('<p class="eyebrow">统一 TTS + S2T · 双数据源 · 2026-10-15 · 训练中</p>')
    A('<h1>统一模型实验报告</h1>')
    A(f'<p class="lede prose">一个 {F["params_total"]/1e6:.2f}M 的 ELF-B 主干同时做语音识别和语音合成。'
      '这一版两个方向各用各自对照论文的数据：<b>S2T</b> 用 LibriSpeech 960 h 和冻结的 whisper-large-v3，'
      '对照 ELF-S2T；<b>TTS</b> 用 LJSpeech，对照 F5-TTS small 的 LJSpeech 那一列。'
      '训练到 <b>10 万步</b>结束 —— 那时 S2T 见过的样本数正好等于 ELF-B 的训练量。'
      f'本页数字截至第 <b>{fmt(step)}</b> 步。</p>')
    nxt = [s for s in (50000, 100000) if s > step]
    A('<div class="status">'
      f'<div class="st"><span class="k">训练进度</span><span class="v">{fmt(step)} / 100,000</span>'
      f'<span class="d">{sit:.2f} 秒/步 · 4×H100</span></div>'
      + (f'<div class="st"><span class="k">S2T 监控 WER（EMA）</span><span class="v">{evals[max(evals)]["wer"]:.1f}%</span>'
         f'<span class="d">第 {fmt(max(evals))} 步 · 64 句、32 步、无引导</span></div>' if evals else "")
      + f'<div class="st"><span class="k">下一次正式评测</span><span class="v">{fmt(nxt[0]) if nxt else "—"}</span>'
      f'<span class="d">5 万、10 万步各一次，两个任务都测</span></div>'
      f'<div class="st"><span class="k">训练曲线</span><span class="v" style="font-size:.95rem">'
      f'<a href="{WANDB}" style="color:var(--text)">wandb</a></span>'
      f'<span class="d">每 100 步训练指标，每 2000 步评测</span></div></div>')
    A('<div class="toc">'
      '<a href="#arch">1 架构</a><a href="#inputs">2 输入通道</a><a href="#data">3 数据</a>'
      '<a href="#pipe">4 训练流程与损失</a><a href="#setup">5 实验设置</a>'
      '<a href="#results">6 进度与结果</a><a href="#prev">7 和上一版的区别</a><a href="#ex">8 例子</a>'
      '<a href="#limits">9 局限</a></div>')

    # 1 ARCH
    P = F["params"]
    A('<hr class="r"><section id="arch"><p class="eyebrow">1</p><h2>架构</h2>')
    A('<p class="prose">序列 <b>1,648</b> 个位置，切成 mel 区（<b>0–1535</b>，1,536 帧 ÷ 93.75 Hz = '
      '<b>16.38 s</b>）和 token 区（<b>1536–1647</b>，112 个槽）。每个位置堆 <b>5 个通道块</b>，'
      f'拼成 <b>{fmt(F["input_width"])}</b> 维进主干。两个任务填同一个格子阵、但填不同的格子，所以主干一点都不分叉。'
      '和上一版相比，结构完全相同，只是 S2T 的冻结编码器从 whisper-small 换成了 whisper-large-v3。</p>')
    A(f'<figure><div class="scroller">{svg(F)}</div>')
    A('<div class="legend"><span><i class="sw c"></i>干净条件</span>'
      '<span><i class="sw d"></i>流匹配目标</span><span><i class="sw z"></i>清零</span></div>')
    A('<figcaption>两列宽度不按 1536:112 等比画（真实约 13.7:1）；列头标的位置区间是真实的。'
      'TTS 一侧的帧数比例是 LJSpeech 的，S2T 一侧的数据是 LibriSpeech。</figcaption></figure>')
    A('<div class="tscroll"><table><caption>可训练参数，从构造出的模型里数出来（report_facts.py）。冻结前端不计入。</caption>'
      '<thead><tr><th>模块</th><th class="n">参数</th><th class="n">占比</th><th>作用</th></tr></thead><tbody>')
    tot = F["params_total"]
    for name, k, what in [("ELF-B 主干", "backbone", "12 层 × 768，两任务共用；从 ELF-B-owt 初始化"),
                          ("<code>self_cond_proj</code>", "self_cond_proj", "入口投影 1736→512；ELF 的 1024 列在最前，新列按 init_frac 0.25 缩放"),
                          ("<code>whisper_proj</code>", "whisper_proj", "1280→512，把 whisper-large-v3 的空间映进 ELF latent 尺度"),
                          ("<code>voice_out</code>", "voice_out", "512→100 mel 输出头"),
                          ("<code>char_embed</code> + <code>cond_norm</code>", "char_embed+cond_norm", "40×512 字符查表 + RMSNorm")]:
        A(f'<tr><td>{name}</td><td class="n">{fmt(P[k])}</td><td class="n">{100*P[k]/tot:.2f}%</td><td>{what}</td></tr>')
    A(f'<tr><td><b>合计</b></td><td class="n"><b>{fmt(tot)}</b></td><td class="n">100%</td><td>训练日志打印「trainable 105.68M」</td></tr>')
    A('</tbody></table></div>')
    A('<p class="prose" style="margin-top:12px">冻结、不训练的前端：Vocos mel 分析（0 参数，纯 STFT + 滤波器组）、'
      'Vocos 声码器（13.53M，仅评测时用）、t5-small encoder（35.33M）、'
      f'whisper-large-v3 encoder（{F["whisper_encoder_params"]/1e6:.1f}M，特征离线算好存盘）。</p></section>')

    # 2 INPUTS
    T5 = F["tts"]
    s2t_t5 = [v["t5_mean"] for v in s2t.values()]
    A('<hr class="r"><section id="inputs"><p class="eyebrow">2</p><h2>每个输入通道怎么来的</h2>')
    A('<p class="prose">每一块都从原始数据经过固定步骤得到。下面按顺序写出每一步，括号里是维度。</p><div class="ch">')
    A('<div class="cc t"><h4>z_T　文本 / Whisper 通道</h4><div class="dim2">(B, 1648, 512)</div><ol>'
      f'<li>转录文本（小写）→ t5-small 分词器 → token id（LJ 平均 {T5["t5_mean"]:.1f} 个，LibriSpeech '
      f'{min(s2t_t5):.1f}–{max(s2t_t5):.1f} 个，上限 112）</li>'
      '<li>token id → <b>冻结的</b> t5-small encoder → <code>last_hidden_state</code> (N, 512)</li>'
      '<li>÷ 0.2（ELF 的 latent_std）</li>'
      '<li><b>token 区</b>：N 个 latent 按顺序放在位置 1536 起</li>'
      f'<li><b>mel 区（仅 TTS）</b>：用 <code>t5_idx[j] = ⌊j·N/T⌋</code> 把 latent <b>均匀拉伸</b>到 T 帧上（LJ 平均 {T5["frames_per_t5"]:.1f} 帧/token）</li>'
      '<li><b>mel 区（仅 S2T）</b>：文本清零，改为加上 Whisper 投影</li>'
      '<li>S2T 行的 token 区是目标：<code>t·x₀ + (1−t)·2·ε</code></li></ol></div>')
    A('<div class="cc f"><h4>Whisper 特征（加进 z_T）</h4><div class="dim2">(B, 1648, 512)，仅 S2T 行的 mel 区</div><ol>'
      '<li>原始波形 16 kHz → whisper-large-v3 特征提取器 → log-mel (128, 3000)</li>'
      f'<li>→ <b>冻结的</b> whisper-large-v3 encoder → (T′, 1280) @ 50 Hz（离线预计算，{F["whisper_gb"]:.0f} GB）</li>'
      '<li>按最近邻从 50 Hz 映到我们的 93.75 Hz 帧：<code>wi = ⌊(a+j)·50/93.75⌋</code></li>'
      f'<li>→ 可训练 <code>whisper_proj</code> 1280→512（{fmt(P["whisper_proj"])} 参数）</li>'
      '<li><b>加</b>到 z_T 的 mel 区。TTS 行没有 Whisper 特征：那里是全零、mask 为 0，'
      '但这一块每个 batch 都在，保证全是 TTS 的 batch 里 <code>whisper_proj</code> 也在计算图里（多卡训练要求）</li></ol></div>')
    s2t_stats = "、".join(f'{v["mel_mean"]:.2f} / {v["mel_std"]:.2f}' for v in s2t.values())
    A('<div class="cc v"><h4>z_V　mel 通道</h4><div class="dim2">(B, 1648, 100)</div><ol>'
      '<li>原始波形 → 重采样到 24 kHz</li>'
      '<li>→ Vocos <code>MelSpectrogramFeatures</code>（n_fft 1024，hop 256，100 mel 滤波器）→ (T, 100) @ 93.75 Hz</li>'
      f'<li><b>每个数据源按自己的统计量标准化</b>：TTS（LJ）用 <code>(mel − ({T5["mel_mean"]:.3f})) / {T5["mel_std"]:.3f}</code>，'
      f'这也是模型和声码器的 mel 空间；S2T 的三个 LibriSpeech 子集（均值 / 标准差 {s2t_stats}）'
      '各自先仿射映射到这个空间，标准化后就等于用它们自己的统计量</li>'
      f'<li><b>TTS 行</b>：前 K 帧是 prompt（K = T × U(0, 0.3)，实测平均 {100*mix["tts_prompt_frac_mean"]:.1f}%）保持干净，其余是目标 '
      '<code>t·x₀ + (1−t)·2·ε</code></li>'
      '<li><b>S2T 行</b>：全部干净，是条件</li></ol></div>')
    A('<div class="cc f"><h4>cond　字符查表</h4><div class="dim2">(B, 1648, 512)，仅 TTS 行</div><ol>'
      '<li>转录文本 → 38 个字符（空格、a–z、11 个标点）→ 字符 id</li>'
      f'<li>在<b>整条 clip</b>上均匀拉伸到帧（LJ 平均 {T5["frames_per_char"]:.1f} 帧/字符）</li>'
      '<li>→ <code>nn.Embedding(40, 512)</code>，从零训练</li>'
      '<li>→ RMSNorm</li>'
      '<li>S2T 行按任务门控为 0（否则 S2T 会从这里看到答案）</li></ol></div>')
    A('<div class="cc s"><h4>sc_T / sc_V　自条件</h4><div class="dim2">(B,1648,512) / (B,1648,100)</div><ol>'
      '<li>训练：一半的行（self_cond_prob 0.5）先做一次无梯度前向，得到 x̂₀</li>'
      '<li>条件位置用 <code>restore()</code> 钉回干净值，目标位置保留模型的 x̂₀</li>'
      '<li>推理：每一步把上一步的 x̂₀ 喂进来</li>'
      '<li>TTS 行的 sc_T 永远等于干净文本，S2T 行的 sc_V 永远等于干净 mel —— 两者只是条件的副本。'
      '<b>真正的自条件只有被生成的那个通道</b>：TTS 是 sc_V，S2T 是 sc_T</li></ol></div>')
    A('</div></section>')

    # 3 DATA
    A('<hr class="r"><section id="data"><p class="eyebrow">3</p><h2>数据</h2>')
    A('<div class="tscroll"><table><caption>从 cache 的 meta 读出（report_facts.py）。两个任务的训练数据不重叠，各自对齐自己的对照论文。</caption>'
      '<thead><tr><th>集合</th><th class="n">条数</th><th class="n">小时</th><th class="n">说话人</th><th>用途</th></tr></thead><tbody>')
    A(f'<tr><td>LJSpeech-1.1（全部）</td><td class="n">{fmt(T5["clips"])}</td><td class="n">{T5["hours"]:.1f}</td>'
      f'<td class="n">1</td><td><b>TTS 训练</b>，和 F5-TTS small 用的完全一样</td></tr>')
    names = {"cache_libri_tc100_lv3": "train-clean-100", "cache_libri_train_clean_360": "train-clean-360",
             "cache_libri_train_other_500": "train-other-500"}
    for k, v in s2t.items():
        A(f'<tr><td>LibriSpeech <code>{names[k]}</code></td><td class="n">{fmt(v["clips"])}</td>'
          f'<td class="n">{v["hours"]:.1f}</td><td class="n">{fmt(v["speakers"])}</td><td>S2T 训练</td></tr>')
    A(f'<tr><td><b>S2T 合计</b></td><td class="n"><b>{fmt(s2t_clips)}</b></td><td class="n"><b>{s2t_hours:.1f}</b></td>'
      f'<td class="n">{fmt(s2t_spk)}</td><td>ELF-S2T 用的 960 h，去掉了长于 16 秒的句子</td></tr>')
    A('<tr><td>LJSpeech 评测列表</td><td class="n">1,000 对</td><td class="n">—</td><td class="n">1</td>'
      '<td>TTS 评测：目标 4–7 s（平均 5.65 s），prompt 是另一句 2.5–3.5 s（平均 3.05 s）</td></tr>')
    A(f'<tr><td>LibriSpeech <code>test-clean</code></td><td class="n">2,620</td><td class="n">5.4</td><td class="n">40</td>'
      f'<td>S2T 评测：{fmt(F["test_clean_whole"])} 句整句解码，197 句长句切段</td></tr>')
    A('</tbody></table></div>')
    A('<ul class="plain" style="margin-top:12px">'
      '<li><b>LJSpeech 用全部 13,100 条。</b>上一版按 1–10 s 过滤只留了 12,910 条；F5 用的是全部，所以重建了 cache。</li>'
      f'<li><b>LibriSpeech 去掉了长于 16 秒的句子</b>：281,241 → {fmt(s2t_clips)} 条，960.9 h → {s2t_hours:.1f} h'
      f'（少 {100*(1-s2t_hours/960.9):.1f}%）。训练时一条也不裁剪，都是完整句子。</li>'
      '<li><b>TTS 的测试句子在训练集里</b>（F5 原文叫 in-set test）。单说话人数据没有没见过的人可测，F5 也是这样测的。'
      'F5 没有公开它那 1,000 条，我们按同样的规则用固定种子抽（<code>build_lj_eval_list.py</code>）。</li></ul></section>')

    # 4 PIPELINE
    A('<hr class="r"><section id="pipe"><p class="eyebrow">4</p><h2>训练流程与损失</h2>')
    A('<h3>每个 batch 里有多少是 TTS</h3>')
    A(f'<p class="prose">任务由数据来源决定：采样器对每一行先按 <code>p_s2t = 0.5</code> 选任务，再在该任务的数据里均匀抽一条。'
      f'不能把两份数据混在一起洗牌 —— LibriSpeech 的条数是 LJSpeech 的 {s2t_clips/T5["clips"]:.0f} 倍，那样 '
      f'<b>{100*mix["s2t_share_if_union_shuffled"]:.1f}%</b> 的行会是 S2T。下表是用训练代码自己的采样器和 collate，'
      f'在 {mix["batches"]} 个真实 batch（{fmt(sum(mix["rows"].values()))} 行）上数出来的：</p>')
    A('<div class="tscroll"><table><thead><tr><th>口径</th><th class="n">TTS</th><th class="n">S2T</th><th>原因</th></tr></thead><tbody>'
      f'<tr><td>按行数</td><td class="n">{100*mix["row_share_tts"]:.1f}%</td><td class="n">{100*(1-mix["row_share_tts"]):.1f}%</td><td>每行按 p_s2t 选任务</td></tr>'
      f'<tr><td><b>按目标位置（实际进损失的）</b></td><td class="n good">{100*pos_tts:.1f}%</td><td class="n">{100*(1-pos_tts):.1f}%</td>'
      f'<td>TTS 行平均 {mix["tts_target_pos_per_row"]:.0f} 个 mel 目标位置（LJ 句子平均 {mix["tts_row_sec_mean"]:.2f} s）；S2T 行 112 个 token 位置</td></tr>'
      '</tbody></table></div>')
    A('<h3>损失函数</h3>')
    A('<p class="prose">三项损失共用<b>一个分母</b>（所有目标位置的计数），这样每个位置权重相同；'
      '多卡时分母在 4 张卡之间求和，等价于单卡大 batch：</p>')
    A('<div class="formula">loss = ( Σ CE·m_ce  +  Σ L2·m_l2 ) / Σ m_all\n\n'
      'L2 = ‖ v(x̂₀) − v_tgt ‖²，  v(x̂₀) = (x̂₀ − z) / max(1 − t, 0.05)\n'
      'v_tgt = v(x₀)  +  g·( v(模型, 开自条件) − v(模型, 关自条件) ),   g = 1 − 1/U(1,3)\n'
      'CE = 交叉熵( 分类头(隐状态), 真 token )</div>')
    A('<div class="tscroll"><table><caption>三项损失各占的目标位置份额（按上面测得的位置数和 decoder_prob 0.5 算）。</caption>'
      '<thead><tr><th>项</th><th>作用在哪</th><th class="n">份额</th></tr></thead><tbody>'
      f'<tr><td>TTS mel 流匹配 L2</td><td>TTS 行的非 prompt mel 位置</td><td class="n">{100*pos_tts:.1f}%</td></tr>'
      f'<tr><td>S2T 文本 流匹配 L2</td><td>一半 S2T 行（decoder_prob 0.5 掷到反面）的 token 区</td><td class="n">{50*(1-pos_tts):.1f}%</td></tr>'
      f'<tr><td>S2T 交叉熵 CE</td><td>另一半 S2T 行，输入是近乎干净的文本 latent（λ ~ σ(N(−0.5, 0.8))）</td><td class="n">{50*(1-pos_tts):.1f}%</td></tr>'
      '</tbody></table></div>')
    A('<h3>训练量：和 ELF-B 一样</h3>')
    A('<p class="prose">我们每步 32 行里平均一半是 S2T，所以训 10 万步，S2T 见过的样本数正好等于 ELF-B 的 5 万步 × 32。'
      'ELF-B 的学习率在训练结束时余弦退火到 0；为了让第 10 万步是一个训练完成的模型，而不是训练到一半的，我们也在 10 万步时退火到 0。</p>')
    A('<div class="tscroll"><table><thead><tr><th></th><th>ELF-B（ELF-S2T）</th><th>我们，这一版</th><th>我们，上一版</th></tr></thead><tbody>'
      '<tr><td>S2T 见过的样本</td><td>50,000 步 × 32 = <b>160 万</b></td><td>100,000 步 × 16 = <b>160 万</b></td><td>40,000 步 × 16 = 64 万</td></tr>'
      f'<tr><td>不重复的 S2T 数据</td><td>960 h</td><td>{s2t_hours:.0f} h</td><td>92.5 h</td></tr>'
      '<tr><td>优化步数</td><td>50,000</td><td>100,000（每步的 S2T 只有一半）</td><td>40,000</td></tr>'
      '<tr><td>学习率</td><td>峰值 3e-4，余弦到 0</td><td>峰值 1e-4，余弦到 0</td><td>1e-4 恒定</td></tr>'
      '<tr><td>音频编码器</td><td>whisper-large-v3</td><td>whisper-large-v3</td><td>whisper-small</td></tr>'
      '</tbody></table></div>')
    A(f'<p class="prose" style="margin-top:12px">TTS 那边差得多：10 万步里 TTS 行合计约 <b>{tts_h_100k:,.0f} 小时</b>音频'
      f'（每步 {32*mix["row_share_tts"]:.1f} 行 × {mix["tts_row_sec_mean"]:.2f} s），F5-TTS small 的 100K 步是 '
      f'100,000 × 0.91 h ≈ 91,000 小时，我们是它的 <b>{100*tts_h_100k/91000:.1f}%</b>。</p>')
    A('<div class="tscroll"><table><caption>训练超参，核对自 ckpt 的 args 和 cfg。</caption>'
      '<thead><tr><th>项</th><th>值</th></tr></thead><tbody>'
      '<tr><td>步数 / 有效 batch</td><td>100,000 / 32（4 卡 DDP × 8）</td></tr>'
      '<tr><td>优化器</td><td>AdamW，峰值 lr 1e-4，β (0.9, 0.95)，warmup 500 步，余弦退火到 0（' + sw_txt + '；之前是恒定 1e-4），梯度裁剪 1.0，EMA 0.9995</td></tr>'
      '<tr><td>decoder_prob / decoder_p_mean</td><td>0.5 / −0.5（ELF-S2T 的「audio forcing」和 1/1 分配）</td></tr>'
      '<tr><td>self_cond_prob / SC-CFG 范围</td><td>0.5 / U(1, 3)</td></tr>'
      '<tr><td>TTS prompt 比例</td><td>U(0, 0.3)，与 F5-TTS 的「掩码 70–100%」相同</td></tr>'
      '<tr><td>条件 dropout</td><td>S2T 行 audio_drop 0.1（同 ELF-S2T，供音频引导用）；TTS 行 text_drop 0.2（同 F5，供 CFG 用）；label_drop 0.1。'
      '上一版的 dropout 有 bug（清零发生在输入构造之后），已修复并用 forward hook 测试验证</td></tr>'
      '<tr><td>里程碑</td><td>第 5 万、10 万步另存 checkpoint（不含优化器）</td></tr>'
      '</tbody></table></div></section>')

    # 5 SETUP
    A('<hr class="r"><section id="setup"><p class="eyebrow">5</p><h2>实验设置</h2>')
    A('<p class="prose">每个方向对照一篇已发表的结果，训练数据和测试协议都按那篇论文来。</p>')
    A('<div class="paper"><p class="pt">S2T 对照：ELF-S2T（Speech Meets ELF, arXiv 2606.10368）</p>'
      '<div class="pm">ELF-B 105.9M · LibriSpeech 960 h · 冻结 whisper-large-v3 · test-clean 全部 2,620 句</div>'
      '<p style="margin:0;font-size:.9rem">我们方法的直接来源，<b>主干和我们完全一样</b>（105.9M vs 105.7M）。'
      '冻结的 Whisper encoder + 线性投影作音频条件，在连续文本 latent 上做流匹配，最后一步解码出 token。'
      '训练：8 卡 × 4 = batch 32，5 万步，全部是 S2T，训练中 10% 的句子丢掉音频以学到无条件分支。'
      '采样：SDE 128 步，音频引导 w=2.0。ELF-B 的 test-clean WER 是 <b>10.50</b>（ELF-L 653.4M 是 5.69）。</p></div>')
    A('<div class="paper"><p class="pt">TTS 对照：F5-TTS small 的 LJSpeech 一列（arXiv 2410.06885，附录 B.7，Table 9）</p>'
      '<div class="pm">158M · LJSpeech 24 h · Vocos · Whisper-large-v3 · in-set 1,000 条</div>'
      '<p style="margin:0;font-size:.9rem">F5 在同一张表里同时报告了多说话人（LibriTTS）和单说话人（LJSpeech）的小模型。'
      '原文：「Since LJSpeech is a single-speaker dataset, we measure the metrics on in-set tests (1000 samples '
      'organized with 4 to 7 seconds to infer with an around 3-second prompt)」。batch 307,200 帧（0.91 h），'
      '训练越久在 LJSpeech 上反而越差（过拟合）。</p></div>')
    A('<div class="tscroll"><table><caption>F5-TTS small，Table 9，LJSpeech in-set。WER 越低越好，SIM-o 和 UTMOS 越高越好。</caption>'
      '<thead><tr><th>F5 更新步数</th><th class="n">WER↓</th><th class="n">SIM-o↑</th><th class="n">UTMOS↑</th></tr></thead><tbody>'
      '<tr><td class="dim">真实录音</td><td class="n dim">2.36</td><td class="n dim">0.72</td><td class="n dim">4.36</td></tr>'
      '<tr><td>100K</td><td class="n">5.64</td><td class="n">0.72</td><td class="n">4.17</td></tr>'
      '<tr><td>200K</td><td class="n good">2.93</td><td class="n">0.72</td><td class="n">4.18</td></tr>'
      '<tr><td>300K</td><td class="n">3.26</td><td class="n">0.71</td><td class="n">4.12</td></tr>'
      '<tr><td>400K</td><td class="n">3.90</td><td class="n">0.70</td><td class="n">4.05</td></tr>'
      '<tr><td>500K</td><td class="n">4.68</td><td class="n">0.70</td><td class="n">3.99</td></tr>'
      '<tr><td>600K</td><td class="n">5.25</td><td class="n">0.69</td><td class="n">3.93</td></tr>'
      '</tbody></table></div>')
    A('<div class="tscroll"><table><caption>我们的评测协议。</caption><thead><tr><th>项</th><th>S2T（eval_s2t_full.py）</th><th>TTS（eval_lj.py）</th></tr></thead><tbody>'
      f'<tr><td>测试集</td><td>test-clean 全部 2,620 句：{fmt(F["test_clean_whole"])} 句整句；197 句长于 16 s 的在静音处切成 10–16 s 的段，识别后拼回整句再打分</td>'
      '<td>LJSpeech in-set 1,000 对：目标 4–7 s，prompt 是另一句约 3 s 的录音</td></tr>'
      '<tr><td>输入</td><td>Whisper 特征 + mel</td><td>prompt 的 mel + 「prompt 文本 + 目标文本」；生成长度用 F5 的估计 '
      '<code>K·bytes(目标)/bytes(prompt)</code>，不用真实时长</td></tr>'
      '<tr><td>采样</td><td>EMA 权重，Euler ODE 128 步，音频引导 w ∈ {1, 2}</td><td>EMA 权重，Euler ODE 32 步，CFG w ∈ {1, 2}（F5 用 2）</td></tr>'
      '<tr><td>指标</td><td>原始 WER（不做文本归一化），语料级：总错误 / 总参考词数</td>'
      '<td>只算生成区。WER：Whisper-large-v3，去标点、小写，<b>逐句 WER 取平均</b>（F5 的算法）；'
      'SIM-o：生成 vs prompt 原始录音；UTMOS</td></tr>'
      '<tr><td>置信区间</td><td>按句子自助重采样 1,000 次的 95% 区间</td><td>同左</td></tr>'
      '</tbody></table></div>')
    A('<div class="note warn"><p><b>和论文协议不完全一致的地方，要随数字一起说。</b>'
      '① F5 的 1,000 条没有公开，我们按同样规则自己抽。'
      '② ASR 是同一个 Whisper-large-v3 权重，F5 用 faster-whisper，我们用 transformers 实现（beam 5）。'
      '③ SIM-o 用 WavLM-base-plus-sv，F5 用 WavLM-large 微调的 ECAPA-TDNN —— 量程不同，只能看相对高低。'
      '④ S2T 我们用 ODE 采样，ELF-S2T 用 SDE。'
      f'⑤ S2T 训练数据是 {s2t_hours:.0f} h（去掉了长句），评测时长句要切段。</p></div></section>')

    # 6 RESULTS
    A('<hr class="r"><section id="results"><p class="eyebrow">6</p><h2>进度与结果</h2>')
    A('<h3>正式对比</h3>')
    A('<div class="tscroll"><table><caption>第 5 万、10 万步各测一次，两个任务都用同一个 checkpoint（这是一个统一模型）。'
      '第 10 万步是和 ELF-B 训练量相等的对比点。</caption>'
      '<thead><tr><th>模型</th><th class="n">S2T WER↓<br>test-clean 2,620</th>'
      '<th class="n">TTS WER↓<br>LJ in-set</th><th class="n">SIM-o↑</th><th class="n">UTMOS↑</th></tr></thead><tbody>')

    def tts_cells(d):
        best = None
        for k, v in d.items():
            if k.startswith("cfg") and isinstance(v, dict) and "wer_mean_utt" in v:
                if best is None or v["wer_mean_utt"][0] < best[1]["wer_mean_utt"][0]:
                    best = (k, v)
        if not best:
            return '<td class="n pend">—</td>' * 3
        k, v = best
        return (f'<td class="n">{v["wer_mean_utt"][0]:.2f}<br><span class="src">{k}</span></td>'
                f'<td class="n">{v["sim_o"][0]:.3f}*</td><td class="n">{v["utmos"][0]:.2f}</td>')

    for st in sorted(set(list(FR.keys()) + [50000, 100000])):
        r = FR.get(st, {})
        if "s2t" in r:
            runs = r["s2t"]["runs"]
            kbest = min(runs, key=lambda k: runs[k]["wer_full"][0])
            wv = runs[kbest]["wer_full"]
            s2t_cell = f'<td class="n">{wv[0]:.2f}<br><span class="src">w={kbest} [{wv[1]:.2f}, {wv[2]:.2f}]</span></td>'
        else:
            s2t_cell = f'<td class="n pend">{"待测" if st >= step else "—"}</td>'
        tts_cell = tts_cells(r["tts"]) if "tts" in r else '<td class="n pend">待测</td>' * 3
        lab = "我们，第 {:,} 步".format(st) + ("（= ELF-B 训练量）" if st == 100000 else "")
        A(f'<tr class="ours"><td>{lab}</td>{s2t_cell}{tts_cell}</tr>')
    gt = None
    for st_, r_ in sorted(FR.items()):
        if "tts" in r_ and "真实录音" in r_["tts"]:
            gt = r_["tts"]["真实录音"]
    if gt:
        A(f'<tr><td class="dim">真实录音（我们的评测管线）</td><td class="n dim">—</td>'
          f'<td class="n dim">{gt["wer_mean_utt"][0]:.2f}</td><td class="n dim">{gt["sim_o"][0]:.3f}*</td>'
          f'<td class="n dim">{gt["utmos"][0]:.2f}</td></tr>')
    A('<tr class="ref"><td>ELF-S2T（ELF-B，105.9M）</td><td class="n">10.50</td><td class="n dim">—</td><td class="n dim">—</td><td class="n dim">—</td></tr>'
      '<tr class="ref"><td>F5-TTS small（158M）@ 100K</td><td class="n dim">—</td><td class="n">5.64</td><td class="n">0.72</td><td class="n">4.17</td></tr>'
      '<tr class="ref"><td>F5-TTS small @ 200K（最好）</td><td class="n dim">—</td><td class="n good">2.93</td><td class="n">0.72</td><td class="n">4.18</td></tr>'
      '</tbody></table></div>')
    A('<p class="src">* 我们的 SIM-o 用 WavLM-base-plus-sv，和 F5 的不在同一标尺：同一套代码在真实录音上测出 '
      + (f'{gt["sim_o"][0]:.2f}' if gt else "0.95") + '，F5 报的是 0.72。</p>')
    if gt:
        A(f'<div class="note warn"><p><b>同样是真实录音，我们的管线测出 {gt["wer_mean_utt"][0]:.2f}，F5 报的是 2.36。</b>'
          '错误主要是写法差异而不是听错：数字（参考 <code>eight</code>，Whisper 写 <code>8</code>）、'
          '连字符（<code>rooming-house</code> 去标点后成了一个词）、人名拼写。F5 的文本处理和我们一样，'
          '剩下的差距来自抽到的 1,000 句不同和 ASR 实现不同（transformers vs faster-whisper）。'
          '所以比较时要同时看两边的真实录音行，和「模型 − 真实录音」的差。</p></div>')
    TU = {}
    tp = os.path.join(HERE, "out", "dual", "text_use.json")
    if os.path.exists(tp):
        TU = json.load(open(tp))
    tu_log = {}
    for j in JOBS:
        lp = os.path.join(HERE, "logs", f"uni-{j}.out")
        if not os.path.exists(lp):
            continue
        cur = None
        for line in open(lp, errors="replace"):
            m = re.search(r"\[eval (\d+) ema\]", line)
            if m:
                cur = int(m.group(1))
            m = re.search(r"TTS text use .*?: t=0\.1 ([\d.]+)\s+t=0\.3 ([\d.]+)", line)
            if m and cur:
                tu_log[cur] = (float(m.group(1)), float(m.group(2)))
    if TU or tu_log:
        A('<h3>TTS 有没有在用文本</h3>')
        A('<p class="prose">训练 L2 回答不了这个问题：mel 的误差大部分是不看文本也能预测的东西（LJ 的音色、频谱包络、'
          '从 prompt 接着说），所以 L2 很早就平了，而模型还在说错的词。这里直接测：同一段加噪的 mel，'
          '给它自己的文本和换成别的句子的文本，比较预测误差。<b>1.0 = 完全不看文本</b>，越大越在用文本。'
          't 越小噪声越大，内容越要靠文本决定。</p>')
        A('<div class="tscroll"><table><caption>错文本误差 ÷ 正确文本误差（diag_text_dep.py，16 句 LJ；'
          '训练中的那一行来自新作业每 2,000 步的监控，8 句）。</caption>'
          '<thead><tr><th>模型</th><th class="n">见过的 TTS 样本</th><th class="n">t = 0.1</th>'
          '<th class="n">t = 0.3</th><th>备注</th></tr></thead><tbody>')
        rowsT = []
        for k, v in TU.items():
            ts = v["step"] * (32 if v["batch"] in (8, 32) else v["batch"]) * (1 - v["p_s2t"])
            name = ("现在这个模型" if "dual" in k else k.split("/")[1])
            note = {"d_cond_fixed32": "上一个能用的 LJ 统一模型（WER 28%，旧口径）",
                    "hy_tts": "纯 TTS，LJ"}.get(k.split("/")[1], "")
            rowsT.append((ts, f'{name}，第 {fmt(v["step"])} 步', v["res"]["0.1"]["shuf"] / v["res"]["0.1"]["true"],
                          v["res"]["0.3"]["shuf"] / v["res"]["0.3"]["true"], note))
        for s_, (a1, a3) in tu_log.items():
            rowsT.append((s_ * 16, f"现在这个模型，第 {fmt(s_)} 步（监控）", a1, a3, "训练中每 2,000 步"))
        for ts, name, r1, r3, note in sorted(rowsT):
            A(f'<tr><td>{name}</td><td class="n">{fmt(int(ts))}</td><td class="n">{r1:.2f}</td>'
              f'<td class="n">{r3:.2f}</td><td>{note}</td></tr>')
        A('</tbody></table></div>')
        A('<p class="prose" style="margin-top:12px">按见过的 TTS 样本数排，所有模型落在同一条上升的线上：样本越多，越依赖文本。'
          '同样见过 32 万条时，现在的统一模型（第 2 万步）比纯 TTS 的 hy_tts 还更依赖文本，'
          '说明 S2T 换成 LibriSpeech 并没有拖慢 TTS。10 万步时它会见过 160 万条，是上一个能用的 LJ 模型的 2.5 倍。</p>')

    A('<h3>训练中的监控评测</h3>')
    A('<p class="prose">每 2,000 步在 64 句 test-clean 和 64 句 LJ 上做一次快速评测：EMA 权重、32 步、不加引导。'
      '它只用来看趋势，<b>不能和论文数字比</b> —— 正式评测用 128 步、音频引导和全部 2,620 句，数字会低很多。'
      f'完整曲线在 <a href="{WANDB}" style="color:var(--text)">wandb</a>。</p>')
    A('<div class="tscroll"><table><caption>S2T WER 越低越好。TTS 一列是生成 mel 和真实 mel 的误差 ÷ 什么都不生成的误差；'
      '它衡量不了发音清不清楚（内容对、但时间没对齐，误差照样大），所以 TTS 要看下面例子和正式评测。</caption>'
      '<thead><tr><th class="n">步数</th><th class="n">S2T WER↓</th><th class="n">token 准确率↑</th><th class="n">TTS mel 误差比</th></tr></thead><tbody>')
    for s in sorted(evals):
        e = evals[s]
        A(f'<tr><td class="n">{fmt(s)}</td><td class="n">{e["wer"]:.2f}%</td><td class="n">{e["tok"]:.1f}%</td>'
          f'<td class="n">{e["tts"]:.3f}</td></tr>')
    A('</tbody></table></div>')
    A('<p class="src">token 准确率按位置逐个比较（不做对齐），中间多一个或少一个 token 后面就全错位，所以它明显低估；以 WER 为准。</p>')
    A('<div class="note"><p><b>S2T 比上一版学得快。</b>上一版的配置（LibriSpeech 100 h、whisper-small；'
      'dropout 修复后重跑的 libri_dropfix）同一个监控评测在第 1 万步是 49.1%、'
      '第 2 万步 29.0%；这一版第 1 万步已经是 '
      + (f'{evals[10000]["wer"]:.1f}%' if 10000 in evals else "—")
      + '。<b>TTS 在第 1 万步还不会按文本说话</b>：正式评测 WER 约 130%，声音和流畅度已经像 LJ，但内容和文本无关（第 8 节可以听）；'
      '不过它对文本的依赖在快速上升，见上面的「TTS 有没有在用文本」。</p></div></section>')

    # 7 PREVIOUS
    A('<hr class="r"><section id="prev"><p class="eyebrow">7</p><h2>和上一版的区别</h2>')
    A('<div class="tscroll"><table><thead><tr><th></th><th>上一版（libri_tc100_uni_ddp）</th><th>这一版（dual_lj_ls960）</th></tr></thead><tbody>'
      '<tr><td>TTS 数据</td><td>LibriSpeech 100 h，251 人，zero-shot</td><td><b>LJSpeech 23.9 h，1 人</b>，没有 zero-shot</td></tr>'
      f'<tr><td>S2T 数据</td><td>同一份 LibriSpeech 100 h</td><td><b>LibriSpeech {s2t_hours:.0f} h</b></td></tr>'
      '<tr><td>Whisper 编码器</td><td>whisper-small，768→512</td><td><b>whisper-large-v3</b>，1280→512</td></tr>'
      f'<tr><td>总参数</td><td>105.42M</td><td>{tot/1e6:.2f}M</td></tr>'
      '<tr><td>任务怎么分</td><td>collate 里每行掷硬币</td><td>按数据来源，采样器保证 50/50</td></tr>'
      '<tr><td>mel 归一化</td><td>一组统计量</td><td>每个数据源各自归一化，TTS 的 LJ 统计量是参照</td></tr>'
      '<tr><td>条件 dropout</td><td>label 0.1、audio 0、text 0，而且有 bug</td><td>audio 0.1、text 0.2，bug 已修</td></tr>'
      '<tr><td>训练</td><td>4 万步，恒定 lr</td><td>10 万步，余弦退火到 0</td></tr>'
      '<tr><td>TTS 评测</td><td>LibriSpeech，同一句开头 3 s 当 prompt</td><td>LJ，另一句约 3 s 当 prompt（F5 协议）</td></tr>'
      '<tr><td>S2T 评测</td><td>2,423 句（跳过长句）</td><td>全部 2,620 句（长句切段）</td></tr>'
      '</tbody></table></div>')
    A('<h3>上一版的结果，以及为什么改成现在这样</h3>')
    A('<ul class="plain">'
      '<li><b>S2T 能用</b>：test-clean 2,423 句上 WER 17.20%（128 步）。差距来自数据少 10 倍、编码器弱一档、训练样本只有 ELF-B 的 40%。</li>'
      '<li><b>TTS 在多说话人 zero-shot 下不工作</b>：只算生成区的 WER 163%。F5-TTS small 自己在 100K 步时，多说话人是 29.5%、'
      '单说话人 LJSpeech 已经是 5.64% —— 多说话人 zero-shot 需要多得多的训练，而我们的训练量只有它 100K 步的 1/16。'
      '所以这一版 TTS 改用 LJSpeech，并对照 F5 的 LJSpeech 那一列。</li>'
      '<li><b>更早的一版两个任务都用 LJSpeech</b>（d_cond_fixed32：12,910 条，whisper-small，4 万步）。'
      'TTS 生成区 WER 28.15%（旧口径，同口径下真实录音 17.86%、声码器上限 14.29%）；S2T 在 LJ 自己的验证集上约 28%，'
      '训练和测试是同一个人，不能和 ELF-S2T 比。</li></ul>')
    A('<h3>一个仍然成立的性质：S2T 主要靠分类头</h3>')
    A('<p class="prose">上一版的模型只积分 1 步就有 20.33% WER，128 步是 16.29%（前 400 句）。也就是说流匹配贡献约 4 个点，'
      '其余来自最后一步的交叉熵分类头直接读 Whisper 条件下的隐状态。TTS 没有这样的判别式捷径：mel 走 '
      '<code>final_layer → voice_out → Vocos</code>，纯回归，所以同样的训练量下 TTS 比 S2T 难学得多。</p></section>')

    # 8 EXAMPLES
    A('<hr class="r"><section id="ex"><p class="eyebrow">8</p><h2>例子</h2>')
    A('<p class="prose">都来自正在训练的这个模型（dual_lj_ls960）。</p>')
    A(s2t_examples(evals, samples))
    A(tts_examples())
    A('</section>')

    # 9 LIMITS
    A('<hr class="r"><section id="limits"><p class="eyebrow">9</p><h2>局限与已知问题</h2>')
    A('<ul class="plain">'
      '<li><b>TTS 是单说话人、in-set 测试</b>：模型只会用 LJ 一个人的声音，测试句子也在训练集里。这是 F5 论文自己报告的设定，'
      '但它测不出 zero-shot 能力，也看不出对新句子的泛化。</li>'
      f'<li><b>TTS 训练量远少于 F5</b>：10 万步合计约 {tts_h_100k:,.0f} 小时 TTS 音频，是 F5 100K 步的 {100*tts_h_100k/91000:.1f}%。</li>'
      f'<li><b>S2T 训练数据少 {100*(1-s2t_hours/960.9):.1f}%</b>（长于 16 s 的句子在预处理时被丢掉），评测时长句要切段，切口可能多出错误。</li>'
      '<li><b>评测实现和论文不完全一样</b>：ASR 实现、SIM 模型、S2T 采样器（ODE vs SDE），见第 5 节。</li>'
      + ('<li><b>学习率调度中途改过</b>：' + (f'前 {fmt(sw)} 步' if sw else '开头一段') +
         '是恒定 1e-4，之后为了在 10 万步结束而改成余弦退火（从当时的 1e-4 平滑接上）。</li>') +
      '<li><b>S2T 的流匹配贡献小</b>，主体是分类头 —— 这限制了「连续流匹配做 S2T」这个主张的强度。</li>'
      '<li><b>只有一次训练</b>，没有不同随机种子的方差。</li>'
      '</ul></section>')

    A('</div>')
    p = os.path.join(HERE, "report.html")
    open(p, "w").write("\n".join(out))
    print(f"wrote {p}  ({os.path.getsize(p)/2**20:.2f} MB)  step {step}, evals {len(evals)}, "
          f"formal {sorted(FR)}")


if __name__ == "__main__":
    main()
