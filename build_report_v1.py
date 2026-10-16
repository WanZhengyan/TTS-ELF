#!/usr/bin/env python
"""Build the experimental report: architecture, inputs, data, pipeline, setup,
results, examples, and one comparison paper per direction.

Every number here was read from a checkpoint, a cache meta, a log, or a
published table during this session; the page names the source next to each.
Audio is embedded from listen_why.json so the TTS numbers can be heard.
"""
import html
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_unified import wer_counts

E = html.escape

CSS = r"""
:root{
  color-scheme: light;
  --ground:#edeff2; --surface:#ffffff; --surface-2:#f4f6f8; --sunk:#e2e6ec;
  --ink:#121820; --ink-2:#3a4552; --muted:#64707d; --line:#ccd4dd;
  --voice:#a8490a; --text:#0a6377; --frozen:#5a3f93; --spk:#1d6b4f;
  --pass:#246a45; --fail:#9b2e26; --warn:#87570a;
  --voice-bg:#fbeee5; --text-bg:#e3f0f3; --frozen-bg:#ece7f6; --spk-bg:#e3f0ea;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
  --sans:"IBM Plex Sans",system-ui,-apple-system,sans-serif;
  --serif:"IBM Plex Serif",Georgia,serif;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme: dark;
    --ground:#0c1015; --surface:#141a22; --surface-2:#1a212a; --sunk:#10161d;
    --ink:#e2e8ef; --ink-2:#bcc7d3; --muted:#8a97a5; --line:#273039;
    --voice:#e89356; --text:#56c0d6; --frozen:#a78ddf; --spk:#5fc294;
    --pass:#62bc8b; --fail:#e7847a; --warn:#d5a547;
    --voice-bg:#271b12; --frozen-bg:#1f1930; --text-bg:#11252b; --spk-bg:#11241b;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --ground:#0c1015; --surface:#141a22; --surface-2:#1a212a; --sunk:#10161d;
  --ink:#e2e8ef; --ink-2:#bcc7d3; --muted:#8a97a5; --line:#273039;
  --voice:#e89356; --text:#56c0d6; --frozen:#a78ddf; --spk:#5fc294;
  --pass:#62bc8b; --fail:#e7847a; --warn:#d5a547;
  --voice-bg:#271b12; --frozen-bg:#1f1930; --text-bg:#11252b; --spk-bg:#11241b;
}
body{background:var(--ground); color:var(--ink); font-family:var(--sans);
     font-size:15px; line-height:1.63; margin:0;}
.wrap{max-width:1020px; margin:0 auto; padding-inline:20px; padding-block:44px 76px;}
.prose{max-width:68ch;}
h1{font-family:var(--serif); font-weight:600; font-size:2.2rem; line-height:1.13;
   letter-spacing:-.015em; margin:0 0 .45rem; text-wrap:balance;}
h2{font-family:var(--serif); font-weight:600; font-size:1.42rem; line-height:1.23;
   margin:0 0 .5rem; text-wrap:balance;}
h3{font-weight:600; font-size:1.03rem; margin:1.4rem 0 .35rem;}
p{margin:0 0 .8rem;}
.lede{font-size:1.07rem; color:var(--ink-2); max-width:64ch;}
.eyebrow{font-family:var(--mono); font-size:.72rem; letter-spacing:.13em;
         text-transform:uppercase; color:var(--muted); margin:0 0 .45rem;}
section{margin-top:50px;}
code{font-family:var(--mono); font-size:.86em; background:var(--surface-2);
     padding:.1em .34em; border-radius:3px;}
hr.r{height:1px; background:var(--line); border:0; margin:50px 0 0;}
.tscroll{overflow-x:auto;}
table{border-collapse:collapse; width:100%; font-size:.872rem;
      font-variant-numeric:tabular-nums; margin:12px 0 0;}
th,td{text-align:left; padding:7px 11px; border-bottom:1px solid var(--line);
      vertical-align:top;}
th{font-family:var(--mono); font-size:.7rem; letter-spacing:.06em;
   text-transform:uppercase; color:var(--muted); font-weight:500;
   border-bottom:1px solid var(--ink-2);}
td.n,th.n{text-align:right;} td.n{font-family:var(--mono);}
tbody tr:hover{background:var(--surface-2);}
caption{caption-side:top; text-align:left; font-size:.85rem; color:var(--muted);
        padding:0 0 7px;}
.good{color:var(--pass); font-weight:600;} .bad{color:var(--fail); font-weight:600;}
.dim{color:var(--muted);}
tr.ours td{background:var(--text-bg); font-weight:500;}
tr.ref td{background:var(--surface-2);}
.src{font-family:var(--mono); font-size:.74rem; color:var(--muted);}
figure{margin:24px 0 0;}
.scroller{overflow-x:auto; border:1px solid var(--line); border-radius:7px;
          background:var(--surface);}
.scroller svg{display:block; max-width:100%; height:auto;}
figcaption{font-size:.85rem; color:var(--muted); margin-top:9px; max-width:74ch;}
.legend{display:flex; gap:17px; flex-wrap:wrap; font-size:.79rem; color:var(--muted);
        margin:10px 0 0; font-family:var(--mono);}
.legend span{display:flex; align-items:center; gap:7px;}
.sw{width:22px; height:12px; border-radius:2px; border:1.5px solid var(--ink-2);}
.sw.d{border-style:dashed; background:transparent;}
.sw.z{background:var(--sunk); border-color:var(--line);}
.sw.c{background:var(--text-bg); border-color:var(--text);}
.note{border-left:3px solid var(--text); background:var(--surface);
      padding:13px 17px; border-radius:0 5px 5px 0; margin:16px 0; font-size:.91rem;}
.note.bad{border-left-color:var(--fail);} .note.warn{border-left-color:var(--warn);}
.note.good{border-left-color:var(--pass);}
.note p:last-child{margin-bottom:0;}
ul.plain{padding-left:1.15rem; margin:.35rem 0 .8rem;} ul.plain li{margin-bottom:.3rem;}
.formula{font-family:var(--mono); font-size:.86rem; background:var(--surface);
         border:1px solid var(--line); border-radius:6px; padding:12px 15px;
         overflow-x:auto; white-space:pre; line-height:1.7; margin:12px 0;}
.ch{display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
    gap:14px; margin:16px 0 0;}
.cc{background:var(--surface); border:1px solid var(--line); border-radius:7px;
    padding:14px 16px; border-top:3px solid var(--line);}
.cc.t{border-top-color:var(--text);} .cc.v{border-top-color:var(--voice);}
.cc.f{border-top-color:var(--frozen);} .cc.s{border-top-color:var(--muted);}
.cc h4{margin:0 0 .2rem; font-size:.98rem;}
.cc .dim2{font-family:var(--mono); font-size:.76rem; color:var(--muted);
          margin-bottom:.5rem;}
.cc ol{margin:.3rem 0 0; padding-left:1.15rem; font-size:.88rem;}
.cc ol li{margin-bottom:.25rem;}
.paper{background:var(--surface); border:1px solid var(--line); border-radius:7px;
       padding:15px 17px; margin:14px 0;}
.paper .pt{font-weight:600; margin:0 0 .2rem;}
.paper .pm{font-family:var(--mono); font-size:.76rem; color:var(--muted);
           margin-bottom:.6rem;}
.ex{margin-top:16px; border:1px solid var(--line); border-radius:8px;
    background:var(--surface); overflow:hidden;}
.ex > header{padding:11px 16px; border-bottom:1px solid var(--line);}
.ex.s2t > header{background:var(--text-bg);} .ex.tts > header{background:var(--voice-bg);}
.ex h4{margin:0; font-size:.98rem;}
.row{padding:13px 16px; border-bottom:1px solid var(--line);}
.row:last-child{border-bottom:0;}
.row .meta{font-family:var(--mono); font-size:.73rem; color:var(--muted);
           margin-bottom:6px;}
dl.pair{display:grid; grid-template-columns:auto 1fr; gap:4px 11px; font-size:.9rem;
        margin:0;}
dl.pair dt{font-family:var(--mono); font-size:.7rem; color:var(--muted);
           text-transform:uppercase; letter-spacing:.05em; padding-top:.2em;}
dl.pair dd{margin:0;}
.players{display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
         gap:10px; margin-top:10px;}
.pl{background:var(--surface-2); border:1px solid var(--line); border-radius:6px;
    padding:8px 10px;}
.pl .lb{font-family:var(--mono); font-size:.69rem; letter-spacing:.05em;
        text-transform:uppercase; color:var(--muted); display:block; margin-bottom:5px;}
.pl audio{width:100%;}
.toc{display:flex; flex-wrap:wrap; gap:6px 16px; font-size:.86rem; margin:14px 0 0;}
.toc a{color:var(--text); text-decoration:none; border-bottom:1px solid var(--line);}
"""

SVG = r"""
<svg viewBox="0 0 1000 738" role="img" aria-label="统一模型架构：五个通道块在 1648 个位置上，TTS 与 S2T 填不同格子，汇入同一个 ELF-B 主干，分出两个输出头">
<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><polygon points="0,1 10,5 0,9" fill="currentColor"/></marker></defs>
<text x="110" y="24" font-family="IBM Plex Sans, sans-serif" font-size="12.5" font-weight="600" fill="currentColor">TTS　文本 → 语音</text>
<text x="556" y="24" font-family="IBM Plex Sans, sans-serif" font-size="12.5" font-weight="600" fill="currentColor">S2T　语音 → 文本</text>
<g font-family="IBM Plex Mono, monospace" font-size="9.5" fill="currentColor" opacity=".6">
<text x="110" y="40">位置 0–1535 · mel 区 · 16.38 s</text><text x="398" y="40">1536–1647</text>
<text x="556" y="40">位置 0–1535 · mel 区</text><text x="844" y="40">1536–1647</text></g>
<g font-family="IBM Plex Mono, monospace" font-size="11" fill="currentColor">
<text x="14" y="66">z_T</text><text x="14" y="78" font-size="9" opacity=".5">512</text>
<text x="14" y="108">sc_T</text><text x="14" y="120" font-size="9" opacity=".5">512</text>
<text x="14" y="150">z_V</text><text x="14" y="162" font-size="9" opacity=".5">100</text>
<text x="14" y="192">sc_V</text><text x="14" y="204" font-size="9" opacity=".5">100</text>
<text x="14" y="234">cond</text><text x="14" y="246" font-size="9" opacity=".5">512</text>
<!-- TTS -->
<rect x="62" y="48" width="326" height="34" rx="3" fill="var(--text-bg)" stroke="var(--text)" stroke-width="1.6"/>
<text x="225" y="69" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="11" fill="var(--text)">T5 latent · 均匀升采样 26.8 帧/token</text>
<rect x="396" y="48" width="56" height="34" rx="3" fill="var(--text-bg)" stroke="var(--text)" stroke-width="1.6"/>
<text x="424" y="69" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="10.5" fill="var(--text)">T5</text>
<rect x="62" y="90" width="326" height="34" rx="3" fill="var(--text-bg)" stroke="var(--text)" stroke-width="1.6" opacity=".5"/>
<text x="225" y="111" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="11" fill="currentColor" opacity=".75">干净文本的副本（restore 钉回，不是自条件）</text>
<rect x="396" y="90" width="56" height="34" rx="3" fill="var(--text-bg)" stroke="var(--text)" stroke-width="1.6" opacity=".5"/>
<rect x="62" y="132" width="326" height="34" rx="3" fill="none" stroke="var(--voice)" stroke-width="2.2" stroke-dasharray="6 4"/>
<text x="225" y="153" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="11" font-weight="600" fill="var(--voice)">mel：前 14% 是 prompt，其余从噪声积分</text>
<rect x="396" y="132" width="56" height="34" rx="3" fill="var(--sunk)" stroke="var(--line)"/>
<text x="424" y="153" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10" fill="currentColor" opacity=".4">0</text>
<rect x="62" y="174" width="326" height="34" rx="3" fill="none" stroke="var(--voice)" stroke-width="1.5" stroke-dasharray="4 4" opacity=".6"/>
<text x="225" y="195" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="11" fill="var(--voice)" opacity=".85">自条件（上一步的 x̂₀）</text>
<rect x="396" y="174" width="56" height="34" rx="3" fill="var(--sunk)" stroke="var(--line)"/>
<text x="424" y="195" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10" fill="currentColor" opacity=".4">0</text>
<rect x="62" y="216" width="326" height="34" rx="3" fill="var(--frozen-bg)" stroke="var(--frozen)" stroke-width="1.6"/>
<text x="225" y="237" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="11" fill="var(--frozen)">字符查表 · 6.6 帧/字符</text>
<rect x="396" y="216" width="56" height="34" rx="3" fill="var(--sunk)" stroke="var(--line)"/>
<text x="424" y="237" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10" fill="currentColor" opacity=".4">0</text>
<!-- S2T -->
<rect x="508" y="48" width="326" height="34" rx="3" fill="var(--frozen-bg)" stroke="var(--frozen)" stroke-width="1.6"/>
<text x="671" y="69" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="11" fill="var(--frozen)">Whisper 投影 768→512（文本已清零）</text>
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
<text x="448" y="288" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="10.5" fill="var(--frozen)">z_T 一行被 T5 和 Whisper 共用，两者占的列不交 —— 实测重叠位置 = 0</text>
<g stroke="currentColor" stroke-width="1.4" fill="none" marker-end="url(#ar)" opacity=".7">
<path d="M 257 300 L 257 318 L 494 318 L 494 334"/><path d="M 703 300 L 703 318 L 506 318 L 506 334"/></g>
<rect x="210" y="338" width="580" height="34" rx="4" fill="var(--surface-2)" stroke="var(--line)"/>
<text x="500" y="360" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="11" fill="currentColor">concat → (B, 1648, 1736)　＝ 512 + 512 + 100 + 100 + 512</text>
<line x1="500" y1="372" x2="500" y2="392" stroke="currentColor" stroke-width="1.4" marker-end="url(#ar)" opacity=".7"/>
<rect x="262" y="396" width="476" height="32" rx="4" fill="var(--surface)" stroke="var(--line)"/>
<text x="500" y="417" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="11" fill="currentColor">self_cond_proj　1736 → 512　（889,344 参数）</text>
<line x1="500" y1="428" x2="500" y2="448" stroke="currentColor" stroke-width="1.4" marker-end="url(#ar)" opacity=".7"/>
<rect x="262" y="452" width="476" height="30" rx="4" fill="var(--surface-2)" stroke="var(--line)" stroke-dasharray="4 3"/>
<text x="500" y="472" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10.5" fill="currentColor" opacity=".8">前置 token：4 时间 + 4 自条件CFG + 4 模型模式 + 4 任务 = 16 位置</text>
<line x1="500" y1="482" x2="500" y2="502" stroke="currentColor" stroke-width="1.4" marker-end="url(#ar)" opacity=".7"/>
<rect x="210" y="506" width="580" height="58" rx="5" fill="var(--surface)" stroke="currentColor" stroke-width="2.2"/>
<text x="500" y="530" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="14" font-weight="600" fill="currentColor">ELF-B 主干　12 层 × 宽 768　104.06M</text>
<text x="500" y="550" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10.5" fill="currentColor" opacity=".7">两任务 100% 共用 · 双向注意力 · 一次混合前向 · RoPE</text>
<g stroke="currentColor" stroke-width="1.4" fill="none" marker-end="url(#ar)" opacity=".7">
<path d="M 330 564 L 330 592"/><path d="M 670 564 L 670 592"/></g>
<rect x="196" y="596" width="268" height="58" rx="4" fill="var(--voice-bg)" stroke="var(--voice)" stroke-width="1.6"/>
<text x="330" y="615" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10.5" fill="var(--voice)">final_layer 768→512 → voice_out 512→100</text>
<text x="330" y="634" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="10.5" fill="var(--voice)" opacity=".85">mel → Vocos → 波形（TTS，回归）</text>
<rect x="536" y="596" width="268" height="58" rx="4" fill="var(--text-bg)" stroke="var(--text)" stroke-width="1.6"/>
<text x="670" y="615" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10.5" fill="var(--text)">proj 768→512 → unembed 512→32,100</text>
<text x="670" y="634" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="10.5" fill="var(--text)" opacity=".85">token 区 argmax → 文本（S2T，分类）</text>
<text x="500" y="686" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="11" fill="currentColor" opacity=".62">任务专属参数只有 task token：2 × 4 × 768 = 6,144，占 105.42M 的 0.0058%</text>
<text x="500" y="708" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="11" fill="currentColor" opacity=".62">注意两个输出头的本质差别：S2T 是交叉熵分类头，TTS 是回归 —— 第 7 节解释为什么这很关键</text>
</svg>
"""


def s2t_examples():
    pairs = [
        ("hello bertie any good in your mind",
         "oh burty any good in your mind"),
        ("number ten fresh nelly is waiting on you good night husband",
         "ten fresh nally is waiting on you good husband"),
        ("after early nightfall the yellow lamps would light up here and there "
         "the squalid quarter of the brothels",
         "after early nightfall the yellow"),
    ]
    H = ['<div class="ex s2t"><header><h4>S2T　语音 → 文本（真实解码）</h4></header>']
    for ref, hyp in pairs:
        c = wer_counts(ref, hyp)
        w = 100.0 * c["err"] / max(c["n"], 1)
        H += [f'<div class="row"><div class="meta">这条 WER {w:.1f}%　·　'
              f'替换 {c["sub"]} / 删除 {c["dele"]} / 插入 {c["ins"]}</div>',
              '<dl class="pair">', f"<dt>参考</dt><dd>{E(ref)}</dd>",
              f"<dt>模型输出</dt><dd>{E(hyp)}</dd>", "</dl></div>"]
    H += ['<div class="row"><p style="margin:0;font-size:.9rem">失败模式：'
          '<b>人名和稀有词</b>（bertie→burty，nelly→nally）和<b>长句提前停住</b>'
          '（第三条在 token 区读到 EOS 就结束了）。内容骨架是对的。</p></div></div>']
    return "\n".join(H)


def tts_examples():
    jp = os.path.join(HERE, "listen_why.json")
    if not os.path.exists(jp):
        return '<div class="note bad"><p>TTS 音频缺失（listen_why.json）。</p></div>'
    d = json.load(open(jp))
    cases = [c for c in d["cases"] if "Libri" in c["corpus"]]
    H = ['<div class="ex tts"><header><h4>TTS　文本 → 语音（真实音频）</h4></header>',
         '<div class="row"><p style="margin:0;font-size:.9rem">'
         '<b>天花板</b>＝真 mel 过同一个声码器，任何生成都超不过它。'
         '<b>仅生成区</b>＝去掉 3 秒 prompt 后模型真正产出的部分，163.06% 打分的就是它。'
         '</p></div>']
    for c in cases:
        H += [f'<div class="row"><div class="meta">{E(c["uid"])}　·　{c["sec"]:.2f}s　·　'
              f'prompt {c["prompt_sec"]:.2f}s　·　生成 {c["sec"]-c["prompt_sec"]:.2f}s</div>',
              '<dl class="pair">', f'<dt>要说的</dt><dd>{E(c["ref"])}</dd>',
              f'<dt>生成区转写</dt><dd>{E(c["hyp_gen_only"]) or "（空）"}</dd>', "</dl>",
              '<div class="players">']
        for lb, k in (("天花板 真mel→声码器", "ceiling"), ("仅生成区", "gen_only")):
            H.append(f'<div class="pl"><span class="lb">{lb}</span><audio controls '
                     f'preload="metadata" src="data:audio/wav;base64,{c[k]}"></audio></div>')
        H.append("</div></div>")
    H += ['<div class="row"><p style="margin:0;font-size:.9rem">'
          '生成区转写由<b>外部的 Whisper-large-v3</b> 给出，不是我们自己的 S2T —— '
          '用自己的 S2T 给自己的 TTS 打分会是循环论证。'
          '音频是语音形状的、音色像那个说话人，但内容和参考无关。</p></div></div>']
    return "\n".join(H)


def main():
    out = []
    A = out.append
    A('<title>统一模型实验报告</title>')
    A('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:'
      'wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:'
      'wght@400;500;600&display=swap">')
    A(f"<style>{CSS}</style>")
    A('<div class="wrap">')
    A('<p class="eyebrow">统一 TTS + S2T · LibriSpeech · 2026-10-15</p>')
    A('<h1>统一模型实验报告</h1>')
    A('<p class="lede prose">一个 105.4M 的 ELF-B 主干同时做语音识别和语音合成。S2T 在 '
      'LibriSpeech test-clean 上达到 17.20% WER；TTS 在多说话人 zero-shot 设置下不工作。'
      '已发表的同规模基线表明，后者主要是训练量不足 —— 我们处在 F5-TTS small 最早检查点的 1/16。</p>')
    A('<div class="toc">'
      '<a href="#arch">1 架构</a><a href="#inputs">2 输入通道</a><a href="#data">3 数据</a>'
      '<a href="#pipe">4 训练流程与损失</a><a href="#setup">5 实验设置</a>'
      '<a href="#results">6 结果</a><a href="#why">7 分析</a><a href="#ex">8 例子</a>'
      '<a href="#limits">9 局限</a></div>')

    # 1 ARCH
    A('<hr class="r"><section id="arch"><p class="eyebrow">1</p><h2>架构</h2>')
    A('<p class="prose">序列 <b>1,648</b> 个位置，切成 mel 区（<b>0–1535</b>，1,536 帧 ÷ 93.75 Hz = '
      '<b>16.38 s</b>）和 token 区（<b>1536–1647</b>，112 个槽）。每个位置堆 <b>5 个通道块</b>，'
      '拼成 <b>1,736</b> 维进主干。两个任务填同一个格子阵、但填不同的格子，所以主干一个字节都不分叉。</p>')
    A(f'<figure><div class="scroller">{SVG}</div>')
    A('<div class="legend"><span><i class="sw c"></i>干净条件</span>'
      '<span><i class="sw d"></i>流匹配目标</span><span><i class="sw z"></i>清零</span>'
      '</div>')
    A('<figcaption>两列宽度不按 1536:112 等比画（真实约 13.7:1）；列头标的位置区间是真实的。'
      '五个通道块合计 <code>input_width 1736</code>，核对自 ckpt 里 <code>self_cond_proj</code> 的形状 (512, 1736)。</figcaption></figure>')
    A('<div class="tscroll"><table><caption>可训练参数，从构造出的模型里数出来。冻结前端不计入。</caption>'
      '<thead><tr><th>模块</th><th class="n">参数</th><th class="n">占比</th><th>作用</th></tr></thead><tbody>')
    rows = [("ELF-B 主干", "104,061,284", "98.71%", "12 层 × 768，两任务共用；从 ELF-B-owt 初始化，加载 98.7%"),
            ("<code>self_cond_proj</code>", "889,344", "0.84%", "入口投影 1736→512；ELF 的 1024 列在最前，新列按 init_frac 0.25 缩放"),
            ("<code>whisper_proj</code>", "393,728", "0.37%", "768→512，把 Whisper 空间映进 ELF latent 尺度"),
            ("<code>voice_out</code>", "51,300", "0.05%", "512→100 mel 输出头"),
            ("<code>char_embed</code> + <code>cond_norm</code>", "20,992", "0.02%", "40×512 字符查表 + RMSNorm"),
            ("<b>合计</b>", "<b>105,416,648</b>", "100%", "核对自训练日志「trainable 105.42M」")]
    for r in rows:
        A(f'<tr><td>{r[0]}</td><td class="n">{r[1]}</td><td class="n">{r[2]}</td><td>{r[3]}</td></tr>')
    A('</tbody></table></div>')
    A('<p class="prose" style="margin-top:12px">冻结、不训练的前端：Vocos mel 分析（0 参数，纯 STFT + 滤波器组）、'
      'Vocos 声码器（13.53M，仅评测）、t5-small encoder（35.33M）、whisper-small encoder（88.2M）。</p></section>')

    # 2 INPUTS
    A('<hr class="r"><section id="inputs"><p class="eyebrow">2</p><h2>每个输入通道怎么来的</h2>')
    A('<p class="prose">每一块都从原始数据经过固定步骤得到。下面按顺序写出每一步，括号里是维度。</p><div class="ch">')
    A('<div class="cc t"><h4>z_T　文本 / Whisper 通道</h4><div class="dim2">(B, 1648, 512)</div><ol>'
      '<li>转录文本 → t5-small 分词器 → token id（平均 44.9 个，上限 112）</li>'
      '<li>token id → <b>冻结的</b> t5-small encoder → <code>last_hidden_state</code> (N, 512)</li>'
      '<li>÷ 0.2（ELF 的 latent_std；encoder 输出 std 0.1585 → 0.79）</li>'
      '<li><b>token 区</b>：N 个 latent 按顺序放在位置 1536 起</li>'
      '<li><b>mel 区（仅 TTS）</b>：用 <code>t5_idx[j] = ⌊j·N/T⌋</code> 把 latent <b>均匀拉伸</b>到 T 帧上</li>'
      '<li><b>mel 区（仅 S2T）</b>：文本清零，改为加上 Whisper 投影（见 sc 下方）</li>'
      '<li>S2T 行的 token 区是目标：<code>t·x₀ + (1−t)·2·ε</code></li></ol></div>')
    A('<div class="cc f"><h4>Whisper 特征（加进 z_T）</h4><div class="dim2">(B, 1648, 512)，仅 S2T 行的 mel 区</div><ol>'
      '<li>原始波形 16 kHz → whisper-small 特征提取器 → log-mel (80, 3000)</li>'
      '<li>→ <b>冻结的</b> whisper-small encoder → (T′, 768) @ 50 Hz（离线预计算，25.5 GB）</li>'
      '<li>按最近邻从 50 Hz 映到我们的 93.75 Hz 帧：<code>wi = ⌊(a+j)·50/93.75⌋</code></li>'
      '<li>→ 可训练 <code>whisper_proj</code> 768→512（393,728 参数）</li>'
      '<li><b>加</b>到 z_T 的 mel 区（那里 S2T 的文本已清零，实测与文本重叠 0 个位置）</li></ol></div>')
    A('<div class="cc v"><h4>z_V　mel 通道</h4><div class="dim2">(B, 1648, 100)</div><ol>'
      '<li>原始波形 → 重采样到 24 kHz</li>'
      '<li>→ Vocos <code>MelSpectrogramFeatures</code>（n_fft 1024，hop 256，100 mel 滤波器）→ (T, 100) @ 93.75 Hz</li>'
      '<li>log 后按全库统计标准化：<code>(mel − (−1.97)) / 2.42</code></li>'
      '<li><b>TTS 行</b>：前 K 帧是 prompt（K = T × U(0, 0.3)，实测平均 14.3%）保持干净，其余是目标 '
      '<code>t·x₀ + (1−t)·2·ε</code></li>'
      '<li><b>S2T 行</b>：全部干净，是条件</li></ol></div>')
    A('<div class="cc f"><h4>cond　字符查表</h4><div class="dim2">(B, 1648, 512)，仅 TTS 行</div><ol>'
      '<li>转录文本 → 38 个字符（空格、a–z、11 个标点）→ 字符 id</li>'
      '<li>在<b>整条 clip</b>上均匀拉伸到帧（6.6 帧/字符），再随音频一起裁剪</li>'
      '<li>→ <code>nn.Embedding(40, 512)</code>，<b>我们从零训练</b>（20,480 参数，初始化 std 0.02）</li>'
      '<li>→ RMSNorm（否则只有 z_T 尺度的 1.14%）</li>'
      '<li>S2T 行按任务门控为 0（泄漏修复）</li></ol></div>')
    A('<div class="cc s"><h4>sc_T / sc_V　自条件</h4><div class="dim2">(B,1648,512) / (B,1648,100)</div><ol>'
      '<li>训练：一半的行（self_cond_prob 0.5）先做一次无梯度前向，得到 x̂₀</li>'
      '<li>条件位置用 <code>restore()</code> 钉回干净值，目标位置保留模型的 x̂₀</li>'
      '<li>推理：每一步把上一步的 x̂₀ 喂进来</li>'
      '<li>TTS 行的 sc_T 永远等于干净文本，S2T 行的 sc_V 永远等于干净 mel —— 两者都只是条件的副本（restore 把条件位置钉回干净值）。<b>真正的自条件只有被生成的那个通道</b>：TTS 是 sc_V，S2T 是 sc_T</li></ol></div>')
    A('</div>')
    A('<div class="note"><p><b>字符查表是我们训出来的，不是从别处来的。</b>证据：LibriSpeech 的转录里括号出现 '
      '<b>0 次</b>，所以 <code>(</code>/<code>)</code> 两行从未收到梯度，余弦相似度停在随机初始化附近 '
      '<b>+0.035</b>；而在有 25 个括号的 LJSpeech 上训的模型里，这一对是 <b>+0.814</b>。'
      '同时它在 LibriSpeech 上自己学出了音素结构：<code>k/p</code> +0.769、<code>b/p</code> +0.696、'
      '<code>m/n</code> +0.587 —— 塞音聚在一起、鼻音聚在一起，没有任何音素监督。</p></div></section>')

    # 3 DATA
    A('<hr class="r"><section id="data"><p class="eyebrow">3</p><h2>数据</h2>')
    A('<div class="tscroll"><table><caption>从 cache 的 meta 读出。</caption>'
      '<thead><tr><th>集合</th><th class="n">条数</th><th class="n">小时</th><th class="n">说话人</th><th>用途</th></tr></thead><tbody>'
      '<tr><td>LibriSpeech <code>train-clean-100</code></td><td class="n">26,752</td><td class="n">92.5</td><td class="n">251</td><td>训练（两个任务共用同一批音频）</td></tr>'
      '<tr><td>LibriSpeech <code>test-clean</code></td><td class="n">2,423</td><td class="n">4.3</td><td class="n">40</td><td>S2T 评测（全部）</td></tr>'
      '<tr><td>&nbsp;&nbsp;其中 4–10 s</td><td class="n">1,237</td><td class="n">—</td><td class="n">40</td><td>TTS 评测（F5-TTS 协议）</td></tr>'
      '</tbody></table></div>')
    A('<ul class="plain" style="margin-top:12px">'
      '<li><b>测试说话人和训练说话人交集为 0</b>（实测），所以 TTS 是真正的 zero-shot：从 3 秒参考音渲染一个从未听过的声音。</li>'
      '<li>每个训练说话人平均只有 <b>0.368 小时</b>。对比 LJSpeech 是单说话人 23.4 小时。</li>'
      '<li>预处理 <code>--max_s 16.0</code> 丢掉了长于 16 秒的句子：训练集 28,539 → 26,752（−6.3%），测试集 2,620 → 2,423（−7.5%）。</li>'
      '<li>mel 区 1,536 帧 ≥ 最长 1,501 帧，所以<b>没有一条被裁剪</b>，训练和评测都是完整句子。</li></ul></section>')

    # 4 PIPELINE
    A('<hr class="r"><section id="pipe"><p class="eyebrow">4</p><h2>训练流程与损失</h2>')
    A('<h3>每个 batch 里有多少是 TTS</h3>')
    A('<p class="prose">collate 对每一行独立掷一次 <code>p_s2t = 0.5</code> 的硬币决定任务。'
      '但两个任务每行贡献的目标位置数差一个数量级，所以<b>按行数和按损失份额是两回事</b>。'
      '下表是在 40 个真实 batch（1,280 行）上数出来的：</p>')
    A('<div class="tscroll"><table><thead><tr><th>口径</th><th class="n">TTS</th><th class="n">S2T</th><th>原因</th></tr></thead><tbody>'
      '<tr><td>按行数</td><td class="n">51.2%</td><td class="n">48.8%</td><td>每行掷硬币</td></tr>'
      '<tr><td><b>按目标位置（实际进损失的）</b></td><td class="n good">90.4%</td><td class="n">9.6%</td>'
      '<td>TTS 行平均 1,005 个 mel 目标位置；S2T 行 112 个 token 位置</td></tr></tbody></table></div>')
    A('<h3>损失函数</h3>')
    A('<p class="prose">三项损失共用<b>一个分母</b>（所有目标位置的计数），这样每个位置权重相同：</p>')
    A('<div class="formula">loss = ( Σ CE·m_ce  +  Σ L2·m_l2 ) / Σ m_all\n\n'
      'L2 = ‖ v(x̂₀) − v_tgt ‖²，  v(x̂₀) = (x̂₀ − z) / max(1 − t, 0.05)\n'
      'v_tgt = v(x₀)  +  g·( v(模型, 开自条件) − v(模型, 关自条件) ),   g = 1 − 1/U(1,3)\n'
      'CE = 交叉熵( 分类头(隐状态), 真 token )</div>')
    A('<div class="tscroll"><table><caption>三项损失各占的目标位置份额（期望值，按上面测得的位置数和 decoder_prob 0.5 算）。</caption>'
      '<thead><tr><th>项</th><th>作用在哪</th><th class="n">份额</th></tr></thead><tbody>'
      '<tr><td>TTS mel 流匹配 L2</td><td>TTS 行的非 prompt mel 位置</td><td class="n">90.4%</td></tr>'
      '<tr><td>S2T 文本 流匹配 L2</td><td>一半 S2T 行（decoder_prob 0.5 掷到反面）的 token 区</td><td class="n">4.8%</td></tr>'
      '<tr><td>S2T 交叉熵 CE</td><td>另一半 S2T 行，输入是近乎干净的文本 latent（λ ~ σ(N(−0.5, 0.8))）</td><td class="n">4.8%</td></tr>'
      '</tbody></table></div>')
    A('<p class="prose" style="margin-top:12px">注意这是<b>位置</b>份额，不是梯度大小：CE 的单位置值（约 1.5–2）比 L2（约 0.4）大。'
      '时间步 <code>t ~ σ(N(−1.5, 0.8))</code>，中位数 0.18，即训练大多发生在高噪声区；噪声尺度 2.0。'
      '这两个值都是 ELF 为文本 latent 调的，没有为 mel 重调过。</p>')
    A('<div class="tscroll"><table><caption>训练超参，核对自 ckpt 的 args 和 cfg。</caption>'
      '<thead><tr><th>项</th><th>值</th></tr></thead><tbody>'
      '<tr><td>步数 / 有效 batch</td><td>40,000 / 32（4 卡 DDP × 8，实测全局 denom 与单卡一致）</td></tr>'
      '<tr><td>优化器</td><td>AdamW，lr 1e-4，β (0.9, 0.95)，warmup 500 步，梯度裁剪 1.0，EMA 0.9995</td></tr>'
      '<tr><td>decoder_prob / decoder_p_mean</td><td>0.5 / −0.5（ELF-S2T 的「audio forcing」）</td></tr>'
      '<tr><td>self_cond_prob / SC-CFG 范围</td><td>0.5 / U(1, 3)</td></tr>'
      '<tr><td>prompt 比例</td><td>U(0, 0.3)，与 F5-TTS 的「掩码 70–100%」相同</td></tr>'
      '<tr><td>label_drop / audio_drop / text_drop</td><td>0.1 / 0.0 / 0.0（见第 9 节：前两者的实现有 bug）</td></tr>'
      '</tbody></table></div></section>')

    # 5 SETUP
    A('<hr class="r"><section id="setup"><p class="eyebrow">5</p><h2>实验设置</h2>')
    A('<p class="prose">每个方向选了一篇<b>用同一个测试集、参数量最接近我们</b>的已发表论文作对照。</p>')
    A('<div class="paper"><p class="pt">S2T 对照：ELF-S2T（Speech Meets ELF, arXiv 2606.10368）</p>'
      '<div class="pm">ELF-B 105.9M · LibriSpeech 960 h · 冻结 Whisper-large-v3 · test-clean</div>'
      '<p style="margin:0;font-size:.9rem">这是我们方法的直接来源，<b>主干规模和我们完全一致</b>（105.9M vs 105.4M）。'
      '它用冻结的 Whisper encoder + 线性投影作音频条件，在连续文本 latent 上做流匹配、最后一步离散化。'
      '采样：SDE 128 步，音频引导 w=2.0。ELF-B 的 test-clean WER 是 <b>10.50</b>（ELF-L 653.4M 是 5.69）。</p></div>')
    A('<div class="paper"><p class="pt">TTS 对照：F5-TTS small（arXiv 2410.06885，附录 B.7，Table 9）</p>'
      '<div class="pm">158M · LibriTTS 585 h（另训一份 LJSpeech 24 h）· Vocos · Whisper-large-v3 · LibriSpeech-PC test-clean</div>'
      '<p style="margin:0;font-size:.9rem">是我们找到的<b>参数量最接近的 zero-shot TTS 结果</b>，而且评测管线和我们<b>完全相同</b>'
      '（Vocos 声码器 + Whisper-large-v3 转写，4–10 s 子集）。它报告了随训练步数变化的整条曲线，'
      '并且同时在多说话人和单说话人数据上训练 —— 正好是我们两个语料的对照。'
      'batch 307,200 帧（0.91 h），每 100K 步约 8 h × 8 张 H100。</p></div>')
    A('<div class="tscroll"><table><caption>我们的评测协议。</caption><thead><tr><th>项</th><th>S2T</th><th>TTS</th></tr></thead><tbody>'
      '<tr><td>测试集</td><td>test-clean 全部 2,423 条</td><td>test-clean 的 4–10 s 子集，1,237 条中取 120</td></tr>'
      '<tr><td>采样</td><td>Euler ODE 128 步，EMA 权重</td><td>Euler ODE 32 步，EMA 权重</td></tr>'
      '<tr><td>条件</td><td>Whisper 特征 + mel</td><td>文本 + <b>3.00 s 绝对</b> prompt（continuation）</td></tr>'
      '<tr><td>指标</td><td>WER（原始），token 准确率</td><td>WER（Whisper-large-v3，<b>只算生成区</b>）、SIM-o、UTMOS</td></tr>'
      '<tr><td>置信区间</td><td>—</td><td>按 clip 自助重采样 1,000 次的 95% 区间</td></tr>'
      '</tbody></table></div>')
    A('<div class="note warn"><p><b>上面的结果用同一份数据训两个任务，所以只对齐了一篇论文。</b>'
      '统一模型里每行掷硬币决定任务，本报告的模型两个任务都用 LibriSpeech train-clean-100。'
      '但两篇论文的训练数据不一样：ELF-S2T 用 <b>LibriSpeech 960 h</b>，F5-TTS small 用 <b>LibriTTS 585 h</b> '
      '—— 后者是从同一批有声书重新切出的 TTS 专用版：按句子切分、保留标点和大小写、去掉噪声片段、<b>原生 24 kHz</b>。'
      '最实际的差别是采样率：LibriSpeech 是 16 kHz，<b>8 kHz 以上没有内容</b>，模型学到的是频带受限的语音。</p>'
      '<p><b>正在准备的对比训练两个方向分开取数据：S2T 行用 LibriSpeech 960 h，TTS 行用 LibriTTS 585 h，主干仍是同一个。</b>'
      '这样两个方向都和各自的论文完全对齐。注意换数据不会单独让 TTS 变好 —— '
      'F5-TTS small 训练用的正是 LibriTTS，在 100K 步时多说话人 WER 仍是 29.5%（见第 7 节）。</p></div>')
    A('<div class="note warn"><p><b>两处与文献协议不完全一致，必须随数字一起说。</b>'
      '① TTS 用的是 continuation（prompt 取同一句开头），文献常引用的是 cross-sentence，Voicebox 两个都报；'
      '② SIM-o 用的是公开的 WavLM-base-plus-sv，F5-TTS 用 WavLM-large 微调的 SV 模型 —— 所以 SIM-o '
      '只能看趋势，不能逐位对比。</p></div></section>')

    # 6 RESULTS
    A('<hr class="r"><section id="results"><p class="eyebrow">6</p><h2>结果</h2>')
    A('<h3>S2T：能用，与同规模基线差 6.7 个点</h3>')
    A('<div class="tscroll"><table><caption>LibriSpeech test-clean。WER <b>越低越好</b>。</caption>'
      '<thead><tr><th>方法</th><th class="n">参数</th><th class="n">训练数据</th><th>冻结 encoder</th><th class="n">WER↓</th></tr></thead><tbody>'
      '<tr class="ours"><td>我们（128 步）</td><td class="n">105.4M</td><td class="n">92.5 h</td><td>whisper-small</td><td class="n">17.20</td></tr>'
      '<tr class="ref"><td>ELF-S2T（ELF-B）</td><td class="n">105.9M</td><td class="n">960 h</td><td>whisper-large-v3</td><td class="n good">10.50</td></tr>'
      '</tbody></table></div>')
    A('<p class="prose" style="margin-top:12px">差距的来源都明确：数据少 10.4 倍、encoder 弱一档（whisper-small 单独跑 '
      'test-clean 是 3.4，large-v3 是 1.9）、以及我们少了 197 条最长的句子（这一项让我们的数字偏乐观）。</p>')
    A('<div class="tscroll"><table><caption>S2T 对流匹配步数的依赖（前 400 条）。</caption>'
      '<thead><tr><th class="n">积分步数</th><th class="n">WER↓</th><th class="n">token 准确率↑</th></tr></thead><tbody>'
      '<tr><td class="n">1</td><td class="n">20.33</td><td class="n">42.59%</td></tr>'
      '<tr><td class="n">2</td><td class="n">28.61</td><td class="n">39.22%</td></tr>'
      '<tr><td class="n">4</td><td class="n">24.42</td><td class="n">39.57%</td></tr>'
      '<tr><td class="n">8</td><td class="n">20.66</td><td class="n">40.23%</td></tr>'
      '<tr><td class="n">32</td><td class="n">17.53</td><td class="n">43.02%</td></tr>'
      '<tr><td class="n">128</td><td class="n good">16.29</td><td class="n">44.60%</td></tr>'
      '</tbody></table></div>')
    A('<div class="note"><p><b>只积分 1 步就有 20.33%，128 步是 16.29%。</b>所以流匹配贡献了约 4 个点，'
      '其余（从「不会」到 20%）来自最后那一步的交叉熵分类头直接读 Whisper 条件下的隐状态。'
      '2 步和 4 步反而比 1 步差，说明少量积分会把 latent 带到分类头没见过的状态。'
      '<b>这对论文的主张有影响</b>：我们的 S2T 大部分是「Whisper 条件下的分类器」，流匹配是改进项而不是主体。</p></div>')

    A('<h3>TTS：不工作，但这与同规模基线在同等训练量下的表现一致</h3>')
    A('<div class="tscroll"><table><caption>LibriSpeech test-clean（PC 版或原版）4–10 s。WER <b>越低越好</b>，SIM-o 和 UTMOS <b>越高越好</b>。'
      '「处理音频」= 更新步数 × 每批音频小时数，是可比的训练量。</caption>'
      '<thead><tr><th>方法</th><th class="n">参数</th><th class="n">训练数据</th><th class="n">处理音频</th><th class="n">WER↓</th><th class="n">SIM-o↑</th><th class="n">UTMOS↑</th></tr></thead><tbody>'
      '<tr class="ours"><td>我们（仅生成区）</td><td class="n">105.4M</td><td class="n">92.5 h</td><td class="n">5,825 h</td><td class="n bad">163.06</td><td class="n">0.823*</td><td class="n">1.742</td></tr>'
      '<tr class="ref"><td>F5-TTS small @ 100K 步</td><td class="n">158M</td><td class="n">585 h</td><td class="n">91,022 h</td><td class="n">29.5</td><td class="n">0.53</td><td class="n">3.78</td></tr>'
      '<tr class="ref"><td>F5-TTS small @ 200K 步</td><td class="n">158M</td><td class="n">585 h</td><td class="n">182,044 h</td><td class="n">4.58</td><td class="n">0.59</td><td class="n">4.07</td></tr>'
      '<tr class="ref"><td>F5-TTS small @ 400K 步</td><td class="n">158M</td><td class="n">585 h</td><td class="n">364,089 h</td><td class="n good">2.44</td><td class="n">0.60</td><td class="n">4.11</td></tr>'
      '<tr><td class="dim">真实录音</td><td class="n dim">—</td><td class="n dim">—</td><td class="n dim">—</td><td class="n dim">1.25 / 2.23†</td><td class="n dim">—</td><td class="n dim">4.24</td></tr>'
      '</tbody></table></div>')
    A('<p class="src">* 我们的 SIM-o 用 WavLM-base-plus-sv，与 F5 的 WavLM-large 不在同一标尺；在我们自己的量程里它处于地板 0.7017（换一个说话人）到天花板 0.9592（真 mel 过声码器）之间的 47%。'
      '† 前者是我们测的真实录音，后者是 F5 报的 LibriSpeech-PC 真实录音。</p>')
    A('<div class="tscroll"><table><caption>我们 TTS 的细节（n=120，3 s prompt）。</caption>'
      '<thead><tr><th>条件</th><th class="n">仅生成区 WER</th><th class="n">95% 区间</th><th class="n">插入 / 替换 / 删除</th><th class="n">SIM-o 位置</th></tr></thead><tbody>'
      '<tr><td>给文本</td><td class="n">163.06</td><td class="n">[125.4, 235.7]</td><td class="n">42% / 55% / 2%</td><td class="n">47%</td></tr>'
      '<tr><td>不给文本</td><td class="n">284.09</td><td class="n">[152.9, 436.1]</td><td class="n">66% / 31% / 3%</td><td class="n">34%</td></tr>'
      '</tbody></table></div>')
    A('<p class="prose" style="margin-top:12px"><b>模型确实在用文本</b>：去掉文本后插入率从 42% 升到 66%，SIM-o 从 47% 降到 34% '
      '（SIM-o 的两个区间 [0.806, 0.839] 和 [0.774, 0.802] 不重叠）。WER 的区间有重叠 —— 在这个失败水平上 WER '
      '太嘈杂，n=120 时区间宽 110 点，没法用来排序设置。</p></section>')

    # 7 WHY
    A('<hr class="r"><section id="why"><p class="eyebrow">7</p><h2>分析：为什么 S2T 行、TTS 不行</h2>')
    A('<h3>最主要的原因：训练量差一个数量级以上</h3>')
    A('<p class="prose">F5-TTS small 的曲线说明了一切。在 <b>100K 步</b>时，它在单说话人 LJSpeech 上已经 '
      '<b>5.64%</b>，在多说话人 LibriTTS 上还是 <b>29.5%</b> —— 多说话人 zero-shot 需要 200K–300K 步。'
      '我们 40,000 步 × 每批 0.146 h = <b>5,825 小时</b>的音频，折合 F5 的 <b>6,400 步</b>，'
      '<b>是它最早报告的那个仍然失败的检查点的 1/16</b>。</p>')
    A('<div class="note good"><p><b>这也回答了「为什么 LJSpeech 行」</b>：F5-TTS small 自己就呈现同样的模式 —— '
      '单说话人很早就会，多说话人要多得多的训练。单说话人只需记住一个声音；zero-shot 要学会「从任意 3 秒推断一个声音」'
      '这个一般能力。所以我们在 LJSpeech 上能工作、在 LibriSpeech 上不能，<b>并不说明架构坏了</b>。</p></div>')
    A('<h3>为什么同样训练不足，S2T 却能用：两个输出头不对称</h3>')
    A('<ul class="plain"><li><b>S2T 有一个判别式捷径。</b>token 来自一个独立的交叉熵分类头 '
      '（<code>elf_tts/model.py:266</code>），直接读主干隐状态，不经过流匹配输出。1 步积分就有 20%，证明它主要靠这个头。'
      '判别任务比生成任务容易学得多。</li>'
      '<li><b>TTS 没有任何东西纠错。</b>mel 走 <code>final_layer → voice_out → Vocos</code>，纯回归，'
      '每一步积分的误差都进最终输出。</li>'
      '<li><b>TTS 的训练目标里有更多模型自己的预测。</b>ELF 的 SC-CFG 锐化项在 TTS 目标里占 15–35%，在 S2T 里只占 4–10%。'
      '在 t=0.75（决定细节的低噪声端）TTS 是 0.33。</li></ul>')
    A('<h3>被实测排除的解释</h3>')
    A('<div class="tscroll"><table><caption>同一架构在能工作的 LJSpeech 和不能工作的 LibriSpeech 上的诊断。都分辨不出成败，所以都不是原因。</caption>'
      '<thead><tr><th>假设</th><th class="n">LJSpeech（行）</th><th class="n">LibriSpeech（不行）</th></tr></thead><tbody>'
      '<tr><td>均匀拉伸的对齐误差</td><td class="n">1.07 词长</td><td class="n">1.04 词长</td></tr>'
      '<tr><td>exposure gap（t=0.75）</td><td class="n">3.63</td><td class="n">3.54</td></tr>'
      '<tr><td>流匹配目标（教师强制误差）</td><td class="n">0.30</td><td class="n">0.26</td></tr>'
      '<tr><td>SC-CFG 项占目标（t=0.75）</td><td class="n">0.346</td><td class="n">0.329</td></tr>'
      '<tr><td>mel 方差中「包络」占比</td><td class="n" colspan="2">mel 18.0% vs 文本 latent 20.0%，几乎一样</td></tr>'
      '<tr><td>TTS 在 batch 里被饿着</td><td class="n" colspan="2">没有：TTS 占 90.4% 的目标位置</td></tr>'
      '</tbody></table></div></section>')

    # 8 EXAMPLES
    A('<hr class="r"><section id="ex"><p class="eyebrow">8</p><h2>例子</h2>')
    A('<p class="prose">都来自产出上表数字的那个模型 <code>libri_tc100_uni_ddp</code>。</p>')
    A(s2t_examples())
    A(tts_examples())
    A('</section>')

    # 9 LIMITS
    A('<hr class="r"><section id="limits"><p class="eyebrow">9</p><h2>局限与已知问题</h2>')
    A('<ul class="plain">'
      '<li><b>训练代码里的条件 dropout 有 bug，本报告的模型受影响。</b><code>audio_drop</code> 和 <code>text_drop</code> '
      '都在主输入构造<b>之后</b>才清零源头，所以模型实际收到的 z_V / z_T 一直是完整的。本报告的模型这两项都设为 0，'
      '所以不受 dropout 本身影响；但 <code>label_drop</code> 只清零了字符表、从未清零 z_T，意味着 TTS 的 CFG 引导在推理时'
      '是向一个训练从未产生过的状态外插（这解释了为什么 w=2.0 让 WER 变差，278 vs 163）。'
      '已修复并用 forward hook 测试验证（修复前精确地在预期的两处失败），修复版正在训练。</li>'
      '<li><b>S2T 只测了 2,423 / 2,620 条</b>，缺的 197 条全长于 16 秒，数字偏乐观。</li>'
      '<li><b>S2T 的流匹配贡献只有约 4 个点</b>，主体是分类头 —— 这限制了「连续流匹配做 S2T」这个主张的强度。</li>'
      '<li><b>TTS 的 WER 在这个水平上太嘈杂</b>（n=120 时区间宽 110 点），该用 SIM-o 的位置和误差构成来比较设置。</li>'
      '<li><b>8 小时预算到不了能比较的 TTS 水平。</b>F5 报告每 100K 步约 64 H100-小时，我们一跑是 24 H100-小时，'
      '连它「多说话人还是 29.5%」的那个点都到不了 —— 要 2.6 倍算力。</li>'
      '</ul></section>')

    A('</div>')
    p = os.path.join(HERE, "report.html")
    open(p, "w").write("\n".join(out))
    print(f"wrote {p}  ({os.path.getsize(p)/2**20:.2f} MB)")


if __name__ == "__main__":
    main()
