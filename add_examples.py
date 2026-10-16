#!/usr/bin/env python
"""Inject a worked-examples section into architecture.html.

The architecture page says what the model is; this says what it produces. The
S2T pairs are real decodes from the run that scored 17.20% on test-clean, and
the TTS audio is the same clips the three-player page uses, so the two numbers
in the results table can be heard and read rather than taken on faith.
"""
import html
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_unified import wer_counts

# Real decodes from libri_tc100_uni_ddp, copied from its evaluation logs.
S2T_PAIRS = [
    ("hello bertie any good in your mind",
     "oh burty any good in your mind"),
    ("number ten fresh nelly is waiting on you good night husband",
     "ten fresh nally is waiting on you good husband"),
    ("after early nightfall the yellow lamps would light up here and there "
     "the squalid quarter of the brothels",
     "after early nightfall the yellow"),
]

EXTRA_CSS = """
  .ex{margin-top:22px; border:1px solid var(--line); border-radius:8px;
      background:var(--surface); overflow:hidden;}
  .ex > header{padding:12px 17px; border-bottom:1px solid var(--line);
               display:flex; align-items:baseline; gap:10px; flex-wrap:wrap;}
  .ex.s2t > header{background:var(--text-bg);}
  .ex.tts > header{background:var(--voice-bg);}
  .ex h3{margin:0;}
  .ex .dirn{font-family:var(--mono); font-size:.68rem; letter-spacing:.06em;
            text-transform:uppercase; padding:.17em .5em; border-radius:3px;}
  .ex.s2t .dirn{background:var(--text); color:var(--surface);}
  .ex.tts .dirn{background:var(--voice); color:var(--surface);}
  .row{padding:14px 17px; border-bottom:1px solid var(--line);}
  .row:last-child{border-bottom:0;}
  .row .meta{font-family:var(--mono); font-size:.74rem; color:var(--muted);
             margin-bottom:6px;}
  dl.pair{display:grid; grid-template-columns:auto 1fr; gap:4px 11px;
          font-size:.9rem; margin:0;}
  dl.pair dt{font-family:var(--mono); font-size:.71rem; color:var(--muted);
             text-transform:uppercase; letter-spacing:.05em; padding-top:.2em;}
  dl.pair dd{margin:0;}
  .wd{font-family:var(--mono); font-size:.72rem; color:var(--muted);
      margin-top:6px;}
  .players{display:grid; grid-template-columns:repeat(auto-fit,minmax(236px,1fr));
           gap:10px; margin-top:11px;}
  .pl{background:var(--surface-2); border:1px solid var(--line); border-radius:6px;
      padding:9px 11px;}
  .pl .lb{font-family:var(--mono); font-size:.7rem; letter-spacing:.05em;
          text-transform:uppercase; color:var(--muted); display:block;
          margin-bottom:5px;}
  .pl audio{width:100%;}
"""


def esc(s):
    return html.escape(str(s))


def main():
    page = os.path.join(HERE, "architecture.html")
    s = open(page).read()
    if "worked-examples" in s:
        sys.exit("已经注入过了，先恢复 architecture.html 再跑")

    H = ['<hr class="r">', '<section id="worked-examples">',
         '<p class="eyebrow">例子</p>',
         "<h2>两个方向实际产出什么</h2>",
         '<p class="prose">都来自那个唯一完整的 LibriSpeech run '
         '<code>libri_tc100_uni_ddp</code>，也就是上表里 S2T 17.20% / TTS 163.06% '
         '那一行。S2T 是真实解码，TTS 是真实音频。</p>']

    # ---------------- S2T
    H += ['<div class="ex s2t"><header><h3>S2T　语音 → 文本</h3>'
          '<span class="dirn">能用</span>'
          '<span class="dim" style="font-family:var(--mono);font-size:.74rem">'
          '条件 = Whisper 特征 + mel；目标 = 48/112 个 token 位置</span>'
          "</header>"]
    for ref, hyp in S2T_PAIRS:
        c = wer_counts(ref, hyp)
        w = 100.0 * c["err"] / max(c["n"], 1)
        H += [f'<div class="row"><div class="meta">这一条的 WER {w:.1f}%'
              f'　·　{c["n"]} 个参考词</div>',
              '<dl class="pair">',
              f"<dt>参考</dt><dd>{esc(ref)}</dd>",
              f"<dt>模型输出</dt><dd>{esc(hyp)}</dd>", "</dl>",
              f'<div class="wd">替换 {c["sub"]}　删除 {c["dele"]}　'
              f'插入 {c["ins"]}</div></div>']
    H += ['<div class="row"><p style="margin:0;font-size:.9rem">'
          '三条合起来说明 S2T 的失败模式：<b>人名和稀有词出错</b>'
          '（bertie→burty，nelly→nally），以及<b>长句会提前停住</b>'
          '（第三条在 token 区读到 EOS 就结束了，丢掉了后半句）。'
          '内容骨架是对的，这是 17% WER 的样子。</p></div>', "</div>"]

    # ---------------- TTS
    cases = []
    jp = os.path.join(HERE, "listen_why.json")
    if os.path.exists(jp):
        d = json.load(open(jp))
        cases = [c for c in d["cases"] if "Libri" in c["corpus"]]
    if cases:
        H += ['<div class="ex tts"><header><h3>TTS　文本 → 语音</h3>'
              '<span class="dirn">不工作</span>'
              '<span class="dim" style="font-family:var(--mono);font-size:.74rem">'
              '条件 = T5 latent + 字符表 + 3s prompt；目标 = 1,536 个 mel 位置'
              "</span></header>",
              '<div class="row"><p style="margin:0;font-size:.9rem">'
              '每条三个播放器。<b>天花板</b>是真 mel 过同一个声码器 —— 任何生成都不可能'
              '超过它，它界定声码器本身的损失。<b>整句</b>的前 3 秒是真录音'
              '（<code>restore()</code> 把 prompt 位置钉死在真实值上），'
              '所以能听到交接处。<b>仅生成区</b>才是 163.06% 那个数字打分的东西。'
              "</p></div>"]
        for c in cases:
            H += [f'<div class="row"><div class="meta">{esc(c["uid"])}　·　'
                  f'{c["sec"]:.2f}s　·　prompt {c["prompt_sec"]:.2f}s　·　'
                  f'生成区 {c["sec"]-c["prompt_sec"]:.2f}s</div>',
                  '<dl class="pair">',
                  f'<dt>要说的话</dt><dd>{esc(c["ref"])}</dd>',
                  f'<dt>生成区转写</dt><dd>{esc(c["hyp_gen_only"]) or "（空）"}'
                  "</dd>", "</dl>", '<div class="players">']
            for lb, key in (("天花板 真mel→声码器", "ceiling"),
                            (f'整句 前{c["prompt_sec"]:.0f}s是真录音', "full"),
                            ("仅生成区 打分的就是这个", "gen_only")):
                H.append(f'<div class="pl"><span class="lb">{esc(lb)}</span>'
                         f'<audio controls preload="metadata" '
                         f'src="data:audio/wav;base64,{c[key]}"></audio></div>')
            H += ["</div></div>"]
        H += ['<div class="row"><p style="margin:0;font-size:.9rem">'
              '听「仅生成区」就能确认那 163% 不是指标口径问题：音频是<b>语音形状的</b>，'
              '音色也像那个说话人（SIM-o 0.8230，地板 0.7017），但说的内容和参考无关。'
              '误差构成是替换 55% + <b>插入 42%</b>、删除只有 2% —— '
              '模型在流畅地乱说，不是沉默。</p></div>', "</div>"]
    else:
        H += ['<div class="note bad"><p>TTS 音频缺失：先跑 '
              '<code>why_lj_works.py</code> 生成 <code>listen_why.json</code>。'
              "</p></div>"]

    H.append("</section>")

    s = s.replace("</style>", EXTRA_CSS + "</style>", 1)
    s = s.rstrip()
    assert s.endswith("</div>"), "页面结尾不是预期的 </div>"
    s = s[: -len("</div>")] + "\n".join(H) + "\n\n</div>\n"
    open(page, "w").write(s)
    print(f"注入完成：S2T {len(S2T_PAIRS)} 例，TTS {len(cases)} 例")
    print(f"architecture.html 现在 {os.path.getsize(page)/2**20:.2f} MB")


if __name__ == "__main__":
    main()
