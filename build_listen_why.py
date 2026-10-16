#!/usr/bin/env python
"""Turn why_lj_works.py's output into a page you can listen to.

Three players per clip, because the interesting thing is the comparison:
ceiling (the true mel through the same vocoder -- the bound), the whole
utterance (whose first seconds are the real recording that restore() pins, so
the hand-off is audible), and the generated span alone, which is what the
numbers in the report actually score.
"""
import html
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

CSS = """
  :root{
    color-scheme: light;
    --ground:#efeef2; --surface:#ffffff; --surface-2:#f6f5f9; --ink:#17151d;
    --ink-2:#403b4a; --muted:#6a6476; --line:#d6d2de;
    --works:#1f6b52; --broken:#9a2f3c; --neutral:#4a4458;
    --works-bg:#e5f0ec; --broken-bg:#f9e9eb;
    --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
    --sans:"IBM Plex Sans",system-ui,-apple-system,sans-serif;
    --serif:"IBM Plex Serif",Georgia,serif;
  }
  @media (prefers-color-scheme: dark){
    :root:not([data-theme="light"]){
      color-scheme: dark;
      --ground:#100e14; --surface:#191620; --surface-2:#201c29; --ink:#eae7f0;
      --ink-2:#c6c0d2; --muted:#938ca2; --line:#2e2938;
      --works:#64c69f; --broken:#e4868f; --neutral:#a79dbd;
      --works-bg:#122019; --broken-bg:#241317;
    }
  }
  :root[data-theme="dark"]{
    color-scheme: dark;
    --ground:#100e14; --surface:#191620; --surface-2:#201c29; --ink:#eae7f0;
    --ink-2:#c6c0d2; --muted:#938ca2; --line:#2e2938;
    --works:#64c69f; --broken:#e4868f; --neutral:#a79dbd;
    --works-bg:#122019; --broken-bg:#241317;
  }
  body{background:var(--ground); color:var(--ink); font-family:var(--sans);
       font-size:15px; line-height:1.62; margin:0;}
  .wrap{max-width:980px; margin:0 auto; padding-inline:20px;
        padding-block:44px 70px;}
  .prose{max-width:66ch;}
  h1{font-family:var(--serif); font-weight:600; font-size:2.2rem; line-height:1.14;
     letter-spacing:-.015em; margin:0 0 .45rem; text-wrap:balance;}
  h2{font-family:var(--serif); font-weight:600; font-size:1.4rem; margin:0 0 .5rem;
     text-wrap:balance;}
  h3{font-weight:600; font-size:1.02rem; margin:0 0 .3rem;}
  p{margin:0 0 .8rem;}
  .lede{font-size:1.06rem; color:var(--ink-2); max-width:62ch;}
  .eyebrow{font-family:var(--mono); font-size:.71rem; letter-spacing:.13em;
           text-transform:uppercase; color:var(--muted); margin:0 0 .45rem;}
  section{margin-top:48px;}
  code{font-family:var(--mono); font-size:.87em; background:var(--surface-2);
       padding:.1em .34em; border-radius:3px;}
  hr.r{height:1px; background:var(--line); border:0; margin:48px 0 0;}
  .tscroll{overflow-x:auto;}
  table{border-collapse:collapse; width:100%; font-size:.875rem;
        font-variant-numeric:tabular-nums; margin:14px 0 0;}
  th,td{text-align:left; padding:7px 11px; border-bottom:1px solid var(--line);
        vertical-align:top;}
  th{font-family:var(--mono); font-size:.71rem; letter-spacing:.06em;
     text-transform:uppercase; color:var(--muted); font-weight:500;
     border-bottom:1px solid var(--ink-2);}
  td.n,th.n{text-align:right;} td.n{font-family:var(--mono);}
  caption{caption-side:top; text-align:left; font-size:.85rem; color:var(--muted);
          padding:0 0 7px;}
  .w{color:var(--works); font-weight:600;} .b{color:var(--broken); font-weight:600;}
  .dim{color:var(--muted);}
  .corpus{margin-top:30px; border:1px solid var(--line); border-radius:8px;
          background:var(--surface); overflow:hidden;}
  .corpus > header{padding:14px 18px; border-bottom:1px solid var(--line);
                   display:flex; align-items:baseline; gap:11px; flex-wrap:wrap;}
  .corpus.ok > header{background:var(--works-bg);}
  .corpus.no > header{background:var(--broken-bg);}
  .corpus h3{margin:0;}
  .badge{font-family:var(--mono); font-size:.68rem; letter-spacing:.06em;
         text-transform:uppercase; padding:.18em .5em; border-radius:3px;}
  .corpus.ok .badge{background:var(--works); color:var(--surface);}
  .corpus.no .badge{background:var(--broken); color:var(--surface);}
  .clip{padding:16px 18px; border-bottom:1px solid var(--line);}
  .clip:last-child{border-bottom:0;}
  .clip .uid{font-family:var(--mono); font-size:.76rem; color:var(--muted);
             margin-bottom:7px;}
  .txt{font-size:.9rem; margin:0 0 4px; display:grid;
       grid-template-columns:auto 1fr; gap:4px 11px;}
  .txt dt{font-family:var(--mono); font-size:.72rem; color:var(--muted);
          text-transform:uppercase; letter-spacing:.05em; padding-top:.18em;}
  .txt dd{margin:0;}
  .players{display:grid; grid-template-columns:repeat(auto-fit,minmax(248px,1fr));
           gap:11px; margin-top:12px;}
  .pl{background:var(--surface-2); border:1px solid var(--line); border-radius:6px;
      padding:9px 11px;}
  .pl .lb{font-family:var(--mono); font-size:.71rem; letter-spacing:.05em;
          text-transform:uppercase; color:var(--muted); display:block;
          margin-bottom:5px;}
  .pl audio{width:100%;}
  .note{border-left:3px solid var(--neutral); background:var(--surface);
        padding:13px 17px; border-radius:0 5px 5px 0; margin:18px 0;
        font-size:.91rem;}
  .note.no{border-left-color:var(--broken);}
  .note.ok{border-left-color:var(--works);}
  .note p:last-child{margin-bottom:0;}
  ul.plain{padding-left:1.15rem; margin:.35rem 0 .8rem;}
  ul.plain li{margin-bottom:.3rem;}
"""


def esc(s):
    return html.escape(str(s))


def players(c):
    rows = [("天花板 真mel→声码器", c["ceiling"]),
            (f"整句 前{c['prompt_sec']:.1f}s是真录音", c["full"]),
            ("仅生成区 报告打分的就是这个", c["gen_only"])]
    out = ['<div class="players">']
    for lb, b64 in rows:
        out.append(f'<div class="pl"><span class="lb">{esc(lb)}</span>'
                   f'<audio controls preload="metadata" '
                   f'src="data:audio/wav;base64,{b64}"></audio></div>')
    out.append('</div>')
    return "".join(out)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "listen_why.json")
    d = json.load(open(src))
    st, al = d["stats"], d.get("align", {})
    # The generator read its stats from whichever cache it sampled audio from,
    # which for LibriSpeech is the TEST cache -- 2423 clips of 40 held-out
    # speakers. The question here is about the TRAINING data, so overwrite those
    # fields from the training caches. Getting this wrong would have compared
    # LJSpeech's training corpus against LibriSpeech's test set.
    import pickle
    TRAIN_CACHE = {"LJSpeech（能工作）": "cache_lj_uni",
                   "LibriSpeech（不工作）": "cache_libri_train_clean_100"}
    for k, cache in TRAIN_CACHE.items():
        if k not in st:
            continue
        ip = os.path.join(HERE, cache, "index.pt")
        if not os.path.exists(ip):
            continue
        m = pickle.load(open(ip, "rb"))
        recs = m["records"]
        hrs = sum(r["dur"] for r in recs) / 3600
        st[k].update(train_cache=cache, speakers=m["n_speakers"],
                     clips=len(recs), hours=hrs,
                     clips_per_speaker=len(recs) / max(m["n_speakers"], 1),
                     hours_per_speaker=hrs / max(m["n_speakers"], 1),
                     frames_per_t5=m["frames_per_t5"])
    keys = list(st.keys())
    H = [f"<title>两个语料的对照试听</title>",
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=IBM+Plex+Serif:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600'
         '&family=IBM+Plex+Mono:wght@400;500&display=swap">',
         f"<style>{CSS}</style>", '<div class="wrap">',
         '<p class="eyebrow">统一 TTS + S2T · 2026-10-14</p>',
         "<h1>两个语料的对照试听</h1>",
         '<p class="lede prose">同一个架构、同一个目标、同一个采样器，在 LJSpeech 上'
         'TTS 能工作，在 LibriSpeech 上完全不工作。差别不在模型，在这两件事'
         '根本不是同一个任务。先听，再看数字。</p>',
         '<hr class="r">']

    # ---- 为什么不是同一个任务
    H += ['<section>', '<p class="eyebrow">根本差别</p>',
          "<h2>一个是背一个声音，一个是从 3 秒渲染没听过的声音</h2>",
          '<div class="tscroll"><table>',
          '<caption>两份<b>训练</b>语料和两次训练的实际配置，从 cache 的 meta 和 ckpt 的 '
          'args 里读出来的。注意说话人数和小时数指的是训练集，不是测试集。</caption>',
          '<thead><tr><th>量</th>'] + \
         [f'<th class="n">{esc(k)}</th>' for k in keys] + ['</tr></thead><tbody>']
    rows = [("说话人数", "speakers", "{:.0f}"),
            ("clips", "clips", "{:.0f}"),
            ("总小时", "hours", "{:.1f}"),
            ("小时 / 说话人", "hours_per_speaker", "{:.3f}"),
            ("mel 帧上限", "mel_frames", "{:.0f}"),
            ("mel 秒数上限", "mel_seconds", "{:.1f}"),
            ("训练 prompt_hi", "prompt_hi_trained", "{:.2f}"),
            ("帧 / t5 token", "frames_per_t5", "{:.1f}")]
    for lbl, k, fmt in rows:
        H.append(f"<tr><td>{esc(lbl)}</td>")
        for kk in keys:
            H.append(f'<td class="n">{fmt.format(st[kk][k])}</td>')
        H.append("</tr>")
    H.append("</tbody></table></div>")

    hps = [st[k]["hours_per_speaker"] for k in keys]
    mfs = [st[k]["mel_frames"] for k in keys]
    H += ['<div class="note no">',
          f'<p><strong>每个说话人的数据量相差 {max(hps)/min(hps):.0f} 倍，'
          f'序列长度相差 {max(mfs)/min(mfs):.1f} 倍。</strong>'
          'LJSpeech 只有一个说话人，而且训练集和测试集是同一个人 —— 模型把那个声音'
          '背下来就行，prompt 是多余的，任务退化成「文本 → 这一个人的声音」。</p>',
          '<p>LibriSpeech 的 test-clean 有 40 个说话人，和 train-clean-100 的 251 个'
          '<b>交集为 0</b>（实测）。所以这是真正的 <b>zero-shot TTS</b>：从 3 秒参考音'
          '渲染一个<b>从未听过</b>的声音，而每个训练说话人只有 22 分钟数据。文献里做这件事的'
          '模型用 6 万到 10 万小时。</p>', '</div>', '</section>', '<hr class="r">']

    # ---- 对齐误差
    if any(al.get(k) for k in keys):
        H += ['<section>', '<p class="eyebrow">条件信号</p>',
              "<h2>均匀拉伸在两个语料上错得一样多</h2>",
              '<p class="prose">TTS 的文本条件是 <code>t5_idx[j] = ⌊j·N/T⌋</code>，'
              '把 token 均匀拉伸到帧上。用 Whisper 的词级时间戳作参考对齐，量它错多少：</p>',
              '<div class="tscroll"><table>',
              '<caption>均匀假设认为第 k 个单元起始于 clip 的 k/N 处；参考值是真实起始时刻。'
              '误差/词长 > 1 表示条件信号指向了错误的词。</caption>',
              '<thead><tr><th>量</th>'] + \
             [f'<th class="n">{esc(k)}</th>' for k in keys] + ['</tr></thead><tbody>']
        arows = [("起始时刻误差 均值", "mean_ms", "{:.0f} ms"),
                 ("中位", "median_ms", "{:.0f} ms"),
                 ("平均词长", "word_ms", "{:.0f} ms"),
                 ("误差 / 词长", "ratio", "{:.2f}"),
                 ("折算 mel 帧", "frames", "{:.0f}"),
                 ("样本词数", "n_words", "{:.0f}")]
        for lbl, k, fmt in arows:
            H.append(f"<tr><td>{esc(lbl)}</td>")
            for kk in keys:
                a = al.get(kk)
                H.append(f'<td class="n">{fmt.format(a[k]) if a else "—"}</td>')
            H.append("</tr>")
        H.append("</tbody></table></div>")
        rr = [al[k]["ratio"] for k in keys if al.get(k)]
        if len(rr) == 2:
            H += ['<div class="note">',
                  f'<p>两边的误差/词长分别是 <b>{rr[0]:.2f}</b> 和 <b>{rr[1]:.2f}</b>。'
                  + ("两者接近，说明<b>错误对齐不是 LJSpeech 能工作的原因</b> —— "
                     "它在两个语料上一样错。所以对齐只是让 TTS 普遍更难，"
                     "真正区分成败的是说话人任务的性质和每说话人的数据量。"
                     if abs(rr[0] - rr[1]) < 0.3 else
                     "两者差距明显，说明均匀拉伸在 LibriSpeech 上额外错得更多，"
                     "是两个语料表现差异的一部分原因。") + '</p>', '</div>']
        H.append("</section>")
        H.append('<hr class="r">')

    # ---- 试听
    H += ['<section>', '<p class="eyebrow">试听</p>',
          "<h2>每条给三个版本</h2>",
          '<p class="prose"><b>天花板</b>是真 mel 过同一个声码器 —— 任何生成都不可能超过它，'
          '它界定了声码器本身的损失。<b>整句</b>的前几秒是真录音（<code>restore()</code> 把 '
          'prompt 位置钉死在真实值上），所以能听到交接处。<b>仅生成区</b>才是报告里打分的东西。</p>']
    for k in keys:
        cs = [c for c in d["cases"] if c["corpus"] == k]
        if not cs:
            continue
        ok = "ok" if "能工作" in k else "no"
        badge = "能工作" if ok == "ok" else "不工作"
        H += [f'<div class="corpus {ok}"><header><h3>{esc(k)}</h3>'
              f'<span class="badge">{badge}</span>'
              f'<span class="dim" style="font-family:var(--mono);font-size:.74rem">'
              f'{esc(st[k]["run"])}</span></header>']
        for c in cs:
            H += [f'<div class="clip"><div class="uid">{esc(c["uid"])} · '
                  f'{c["sec"]:.2f}s · prompt {c["prompt_sec"]:.2f}s</div>',
                  '<dl class="txt">',
                  f'<dt>参考</dt><dd>{esc(c["ref"])}</dd>',
                  f'<dt>仅生成区的转写</dt><dd>{esc(c["hyp_gen_only"]) or "<空>"}</dd>',
                  "</dl>", players(c), "</div>"]
        H.append("</div>")
    H.append("</section>")

    H += ['<hr class="r">', '<section>', '<p class="eyebrow">结论</p>',
          "<h2>三个原因，按确定程度排序</h2>", '<ul class="plain">',
          '<li><b>任务性质不同（已实测确认）。</b>LJSpeech 训练测试同一个说话人，'
          '模型背下那个声音即可；LibriSpeech 的测试说话人和训练集交集为 0，是真 zero-shot。'
          '这两件事的难度不在一个量级。</li>',
          f'<li><b>每说话人数据量差 {max(hps)/min(hps):.0f} 倍（已实测）。</b>'
          'LJSpeech 一个声音有 23.4 小时；LibriSpeech 每个声音只有 22 分钟，'
          '而模型要学的是「从任意 3 秒推断一个声音」这个一般能力。</li>',
          '<li><b>训练的 prompt 范围不匹配（已实测）。</b>两边都用 <code>prompt_hi=0.3</code>。'
          '在 LJSpeech 上无所谓（prompt 多余），在 LibriSpeech 上致命：prompt 是唯一的'
          '说话人信息，而文献协议要 3 秒 —— 在 4 秒的 clip 上是 75%，完全在训练分布之外。'
          '这是下一跑必须先改的一条。</li>', '</ul>',
          '<div class="note no"><p>所以「小数据集能工作」其实是「更简单的任务能工作」。'
          '把 LJSpeech 的成功读成「模型会做 TTS」是过度解读 —— 它证明的是模型能背一个声音，'
          '不是能从 3 秒渲染一个新声音。</p></div>',
          "</section>", "</div>"]

    out = os.path.join(HERE, "listen_why.html")
    open(out, "w").write("\n".join(H))
    print(f"wrote {out}  ({os.path.getsize(out)/2**20:.2f} MB)")


if __name__ == "__main__":
    main()
