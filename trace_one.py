#!/usr/bin/env python
"""Walk ONE real clip through every named module and print the actual shapes.

Averages do not tell you how anything works. This takes a single clip, names
each module, says in one line what it does, and prints the exact shape going in
and coming out -- the real numbers for this one clip, not a batch mean.
"""
import os, pickle, sys

import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

W = 74
def step(n, name, what, shape_in, shape_out, extra=""):
    print(f"\n[{n}] {name}")
    print(f"     做什么: {what}")
    print(f"     {shape_in}  ->  {shape_out}")
    if extra:
        for line in extra.split("\n"):
            print(f"     {line}")


@torch.no_grad()
def main():
    dev = torch.device("cpu")
    meta = pickle.load(open(os.path.join(HERE, "cache_lj_uni", "index.pt"), "rb"))
    rec = next(r for r in meta["records"] if r["T"] > 560)   # 一条够长的
    uid = rec["id"]
    id2text = {}
    with open(os.path.join(HERE, "data", "LJSpeech-1.1", "metadata.csv"),
              encoding="utf-8") as f:
        for line in f:
            p = line.strip().split("|")
            if len(p) >= 3:
                id2text[p[0]] = p[2]
    text = id2text[uid]
    print("=" * W)
    print(f"这一条: {uid}")
    print(f"文本: {text[:68]}")
    print("=" * W)

    # ---------- 1 原始波形
    wav, sr = torchaudio.load(os.path.join(HERE, "data", "LJSpeech-1.1",
                                           "wavs", uid + ".wav"))
    step(1, "磁盘上的 .wav", "就是一串采样点，没有任何处理",
         f"文件", f"(1, {wav.shape[-1]})  @ {sr} Hz",
         f"1 = 单声道,  {wav.shape[-1]} = 采样点个数 = {wav.shape[-1]/sr:.2f} 秒")

    w24 = torchaudio.functional.resample(wav, sr, 24000)
    step(2, "重采样", "把 22050 Hz 改成 24000 Hz，因为 Vocos 只认 24 kHz",
         f"(1, {wav.shape[-1]})", f"(1, {w24.shape[-1]})")

    # ---------- 2 Vocos 分析侧
    from vocos.feature_extractors import MelSpectrogramFeatures
    fe = MelSpectrogramFeatures(sample_rate=24000, n_fft=1024, hop_length=256,
                                n_mels=100, padding="center")
    mel = fe(w24)[0].transpose(0, 1)
    step(3, "Vocos MelSpectrogramFeatures（分析侧，0 参数）",
         "每 256 个采样点切一帧，对每帧做 FFT，再把频率压成 100 个 mel 频带取 log",
         f"(1, {w24.shape[-1]})", f"({mel.shape[0]}, {mel.shape[1]})",
         f"{mel.shape[0]} 帧 = {w24.shape[-1]} / 256,  每帧 100 个数\n"
         f"第 0 维最低频, 第 99 维最高频")

    # ---------- 3 裁剪 + 归一化
    T = 512
    a = 0
    mel_c = mel[a:a + T]
    step(4, "随机裁剪", f"取连续 512 帧（音频区容量）。这条有 {mel.shape[0]} 帧，所以要裁",
         f"({mel.shape[0]}, 100)", f"({T}, 100)",
         f"起点 a = {a},  512 帧 = {T/93.75:.2f} 秒")
    mu, sd = meta["latent_mean"], meta["latent_std"]
    x0v = (mel_c - mu) / sd
    step(5, "归一化", f"减全局均值 {mu:.4f} 再除标准差 {sd:.4f}",
         f"({T}, 100)", f"({T}, 100)",
         f"结果 std = {x0v.std():.3f}  <- 这就是语音通道的 x0_V")

    # ---------- 4 文本
    from transformers import AutoTokenizer, T5EncoderModel
    tk = AutoTokenizer.from_pretrained("t5-small")
    # 转录按音频裁剪同步裁
    nch = len(rec["chars"])
    c0 = (a * nch) // mel.shape[0]
    c1 = max(c0 + 1, ((a + T) * nch) // mel.shape[0])
    itos = {v: k for k, v in meta["stoi"].items()}
    sub = "".join(itos.get(int(c), "") for c in rec["chars"][c0:c1])
    if c1 < nch and " " in sub:
        sub = sub[:sub.rindex(" ")]
    sub = sub.strip()
    ids = tk(sub).input_ids
    N = len(ids)
    step(6, "T5 分词器", "把字符串切成 subword，每个给一个整数 id",
         f'"{sub[:46]}..."', f"({N},) 个整数",
         f"N = {N} 个 token（这一条的真实值，不是均值）\n"
         f"前几个: {ids[:6]}  = {[tk.decode([i]) for i in ids[:4]]}")

    t5 = T5EncoderModel.from_pretrained("t5-small").eval()
    h = t5(input_ids=torch.tensor([ids])).last_hidden_state[0] / 0.2
    step(7, "t5-small encoder（冻结 35.33M）",
         "每个 token 变成一个 512 维向量，而且带上下文（同一个词在不同句子里不一样）",
         f"({N},) 整数", f"({N}, 512)",
         f"除以 latent_std 0.2 后 std = {h.std():.3f}\n"
         f"512 = t5-small 的 d_model")

    up = np.minimum((np.arange(T) * N) // T, N - 1)
    x0t_audio = h[up]
    step(8, "升采样（短变长，整段复制）",
         f"第 j 帧去读第 ⌊j·{N}/{T}⌋ 个 token 的向量",
         f"({N}, 512)", f"({T}, 512)",
         f"每个 token 连续占 {T//N}–{T//N+1} 帧\n"
         f"帧 0–{int((up==0).sum())-1} 全拿 '{tk.decode([ids[0]])}' 那同一个向量\n"
         f"所以这 512 行里只有 {N} 个不同的向量")

    # ---------- 5 Whisper
    from transformers import WhisperFeatureExtractor, WhisperModel
    wfe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    w16 = torchaudio.functional.resample(wav, sr, 16000)[0]
    feats = wfe(w16.numpy(), sampling_rate=16000, return_tensors="pt").input_features
    step(9, "Whisper 的特征提取器", "它自己的 mel，80 个频带，而且补到固定 30 秒",
         f"(1, {w16.shape[0]}) @16 kHz", f"{tuple(feats.shape)}",
         f"80 = Whisper 内部的频带数（和我们的 100 无关）\n"
         f"3000 = 30 秒 × 100 帧/秒，输入多长都是这个形状")
    wenc = WhisperModel.from_pretrained("openai/whisper-small").encoder.eval()
    wh_all = wenc(feats).last_hidden_state[0]
    M = w16.shape[0] // 160 // 2
    wh = wh_all[:M]
    step(10, "Whisper encoder（冻结 88.2M，不加载它的 decoder）",
         "把声音编成「音素内容接近线性可读」的表示",
         f"{tuple(feats.shape)}", f"({M}, 768)",
         f"输出总是 1500 帧，只有前 {M} 帧是真的（{w16.shape[0]}/160/2）\n"
         f"768 = whisper-small 的 d_model,  帧率 50 Hz")

    wi = np.minimum(((a + np.arange(T)) * (50 / 93.75)).astype(int), M - 1)
    wh_frames = wh[wi]
    step(11, "重采样到我们的帧率", "Whisper 是 50 Hz、我们是 93.75 Hz，取最近邻",
         f"({M}, 768)", f"({T}, 768)",
         f"第 j 帧读第 ⌊j × {50/93.75:.4f}⌋ 个 Whisper 帧")

    # ---------- 6 模型
    from unified import UnifiedELF
    mod = UnifiedELF("ELF-B", char_vocab=meta["char_vocab_size"],
                     max_length=560, whisper_dim=768).eval()
    L = 560
    proj = mod.whisper_proj(wh_frames)
    step(12, "whisper_proj（可训练，393,728 参数）",
         "把 Whisper 的向量空间线性映进「文本通道」的 512 维",
         f"({T}, 768)", f"({T}, 512)",
         f"768 -> 512。不是把音频变成文本，只是借这个通道进主干")

    ch_full = rec["chars"][np.minimum((np.arange(mel.shape[0]) * nch)
                                      // mel.shape[0], nch - 1)]
    chars = torch.zeros(L, dtype=torch.long)
    chars[:T] = torch.from_numpy(ch_full[a:a + T].astype(np.int64))
    cond = mod.cond_norm(mod.char_embed(chars).float())
    step(13, "char_embed + RMSNorm（20,480 + 512 参数）",
         "逐帧的字符 id 查表成向量，再归一化到和 z 同量级",
         f"({L},) 整数（词表 40）", f"({L}, 512)")

    # 拼起来
    z_T = torch.zeros(L, 512); z_V = torch.zeros(L, 100)
    z_T[:T] = x0t_audio.float()            # TTS: 这里放升采样文本
    z_T[512:512 + N] = h.float()           # token 区放原生 latent
    z_V[:T] = x0v.float()
    sc_T = torch.zeros(L, 512); sc_V = torch.zeros(L, 100)
    x = torch.cat([z_T, sc_T, z_V, sc_V, cond], dim=-1)
    step(14, "拼接五个通道块", "每个位置把五块接成一个长向量",
         "z_T 512 + sc_T 512 + z_V 100 + sc_V 100 + cond 512",
         f"({L}, 1736)",
         f"{L} = 音频区 512 + token 区 48")

    e = mod.elf
    y = e.self_cond_proj(x)
    step(15, "self_cond_proj（889,344 参数，ELF 预训练）",
         "把 1736 压回 512。前 1024 列装的是 ELF 原来的权重",
         f"({L}, 1736)", f"({L}, {y.shape[-1]})")
    y1 = e.text_proj.proj1(y)
    step(16, "text_proj.proj1（65,536 参数，ELF 预训练）",
         "先压到 128 —— ELF 自己设计的瓶颈",
         f"({L}, 512)", f"({L}, {y1.shape[-1]})")
    y2 = e.text_proj.proj2(y1)
    step(17, "text_proj.proj2（99,072 参数，ELF 预训练）", "再升到主干宽度 768",
         f"({L}, 128)", f"({L}, {y2.shape[-1]})")
    step(18, "拼 16 个 prefix token", "4 time + 4 sc-cfg + 4 task + 4 mode，拼在最前面",
         f"({L}, 768)", f"({L + 16}, 768)",
         "task 那 4 个按任务选，这是唯一按任务分叉的参数（6,144 个）")
    step(19, "12 × ELFBlock（85,049,856 参数，ELF 预训练）",
         "双向自注意力 + MLP，重复 12 次。两个任务走的是同一份权重、同一个 mask",
         f"({L + 16}, 768)", f"({L + 16}, 768)",
         "12 头 × 每头 64 维。这里占全模型的 81%")
    step(20, "切掉 prefix", "前面拼的 16 个不要了",
         f"({L + 16}, 768)", f"({L}, 768)")

    hid = y2  # 形状相同，用来演示出口的形状
    o1 = e.final_layer(hid)
    step(21, "final_layer（394,496 参数，ELF 预训练）", "denoiser 的输出层",
         f"({L}, 768)", f"({L}, {o1.shape[-1]})")
    o2 = mod.voice_out(o1)
    step(22, "voice_out（51,300 参数，新增）  ← TTS 的出口",
         "512 变 100，也就是预测出的 mel",
         f"({L}, 512)", f"({L}, {o2.shape[-1]})")
    hh = torch.nn.functional.gelu(hid @ e.proj_kernel + e.proj_bias)
    lg = hh @ e.unembed_kernel + e.unembed_bias
    step(23, "proj_kernel + unembed（393,728 + 16,467,300，ELF 预训练）  ← S2T 的出口",
         "768 → 512 → 32100。每个位置给整个 T5 词表打分",
         f"({L}, 768)", f"({L}, {lg.shape[-1]})",
         "只取 token 区那 48 行，argmax 得 token id，截到第一个 EOS")

    from vocos import Vocos
    voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval()
    wav_out = voc.decode(mel_c.unsqueeze(0).transpose(1, 2))
    step(24, "Vocos 声码器（13.53M，冻结）  ← 只在评测时用",
         "把 mel 变回波形。这一步完全不经过我们的模型",
         f"({T}, 100)", f"{tuple(wav_out.shape)}",
         f"{wav_out.shape[-1]} = 512 × 256 = {wav_out.shape[-1]/24000:.2f} 秒")
    print("\n" + "=" * W)
    print(f"这一条的具体数字: 音频 {mel.shape[0]} 帧裁到 {T} 帧,  文本 {N} 个 token")
    print(f"TTS 的 flow 目标 = {T} 个位置（这条没给 voice prompt）")
    print(f"S2T 的 flow 目标 = 48 个位置（固定，与 N={N} 无关）")
    print("=" * W)


if __name__ == "__main__":
    main()
