#!/usr/bin/env python
"""Print every number in the reference document next to where it comes from.

A dimension dropped into a document is not checkable. Each one here is tagged
by what KIND of number it is, because that determines whether it can be changed:

  FIXED      an external dependency's constant. Changing it means changing the
             dependency (a different vocoder, a different T5).
  ELF        from ELF's own config.yml or model definition. Changing it departs
             from the pretrained checkpoint.
  OURS       a value we picked. Free to change.
  DERIVED    arithmetic on the above. Shown with the arithmetic.
  MEASURED   printed from the instantiated model, a real batch, or the cache.
             Shown with the code path that produced it.
"""
import os, pickle, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ROWS = []


def row(kind, name, value, how):
    ROWS.append((kind, name, str(value), how))


def main():
    # ---------------- FIXED: external constants
    from vocos.feature_extractors import MelSpectrogramFeatures
    fe = MelSpectrogramFeatures(sample_rate=24000, n_fft=1024, hop_length=256,
                                n_mels=100, padding="center")
    row("FIXED", "mel 频带数", 100,
        "Vocos charactr/vocos-mel-24khz 的 n_mels。声码器只认这 100 个，"
        "换数字就得换声码器")
    row("FIXED", "mel hop", 256, "同上，Vocos 的 hop_length")
    row("FIXED", "mel 采样率", 24000, "同上，Vocos 的 sample_rate")

    from transformers import AutoConfig
    t5 = AutoConfig.from_pretrained("t5-small")
    row("FIXED", "文本通道宽度", t5.d_model,
        f"t5-small 的 d_model = {t5.d_model}，AutoConfig 读出")
    row("FIXED", "词表大小", 32100,
        "ELF 的 vocab_size，等于 t5 的词表。decode head 的输出宽度")
    wc = AutoConfig.from_pretrained("openai/whisper-small")
    row("FIXED", "Whisper 特征宽度", wc.d_model,
        f"openai/whisper-small 的 d_model = {wc.d_model}")
    row("FIXED", "Whisper hop / stride", "160 / 2",
        "WhisperFeatureExtractor 的 hop_length=160，encoder 的 conv stride=2")

    # ---------------- ELF: from its config and model
    import yaml
    cfgp = os.path.join(HERE, "ckpt", "ELF-B-owt-torch", "config.yml")
    ec = yaml.safe_load(open(cfgp)) if os.path.exists(cfgp) else {}
    for k, label in (("bottleneck_dim", "text_proj 瓶颈宽度"),
                     ("num_time_tokens", "time token 数"),
                     ("num_self_cond_cfg_tokens", "sc-cfg token 数"),
                     ("num_model_mode_tokens", "mode token 数"),
                     ("latent_std", "文本 latent 的尺度"),
                     ("denoiser_p_mean", "denoiser 的 t 分布中心"),
                     ("denoiser_p_std", "denoiser 的 t 分布宽度"),
                     ("denoiser_noise_scale", "denoiser 噪声尺度"),
                     ("decoder_noise_scale", "decoder 噪声尺度"),
                     ("decoder_prob", "decoder 分支概率（ELF 默认）"),
                     ("self_cond_prob", "自条件保留概率"),
                     ("t_eps", "1−t 的钳位下限")):
        if k in ec:
            row("ELF", label, ec[k], f"ELF-B-owt 的 config.yml：{k}")

    from elf_tts.model import ELF_models
    row("ELF", "主干宽度", 768, "ELF_models['ELF-B'] 的 hidden_size")
    row("ELF", "层数 / 头数", "12 / 12", "同上 depth / num_heads")
    row("DERIVED", "每头维度", 768 // 12, "768 / 12 = 64")

    # ---------------- OURS
    row("OURS", "序列长度 L", 560, "我们选的 max_length。RoPE 表按它建")
    row("OURS", "token 区长度", 48,
        "我们选的 max_text，取 t5 token 数的 p99 之上（见下面 MEASURED）")
    row("OURS", "audio forcing 的值", -0.5,
        "ELF-S2T 论文的 decoder_p_mean（原文 −0.5，ELF 默认 0.8）")
    row("OURS", "decoder_prob（最好那跑）", 0.5,
        "ELF-S2T 论文的 \"1/1 split\"")
    row("OURS", "batch / lr / 步数", "32 / 1e-4 / 40000", "我们选的")
    row("OURS", "推理步数 NFE", 32, "我们选的，均匀网格")

    # ---------------- DERIVED
    fr = 24000 / 256
    row("DERIVED", "mel 帧率", f"{fr:.2f} Hz", "24000 / 256 = 93.75")
    row("DERIVED", "音频区帧数", 560 - 48, "max_length − max_text = 512")
    row("DERIVED", "音频长度上限", f"{512/fr:.2f} 秒", f"512 / {fr:.2f} = 5.46")
    row("DERIVED", "Whisper 帧率", f"{16000/160/2:.0f} Hz",
        "16000 / 160 / 2 = 50")
    row("DERIVED", "Whisper→mel 索引", "⌊j × 0.5333⌋",
        f"50 / {fr:.2f} = 0.5333")
    row("DERIVED", "输入拼接宽度", 2 * 512 + 2 * 100 + 512,
        "z_T 512 + sc_T 512 + z_V 100 + sc_V 100 + cond 512 = 1736")
    row("DERIVED", "prefix token 总数", 4 + 4 + 4 + 4,
        "time 4 + sc-cfg 4 + task 4 + mode 4 = 16")
    row("DERIVED", "波形输出长度", 512 * 256,
        "512 帧 × hop 256 = 131072 采样点")

    # ---------------- MEASURED: the cache
    m = pickle.load(open(os.path.join(HERE, "cache_lj_uni", "index.pt"), "rb"))
    Ts = np.array([r["T"] for r in m["records"]])
    Ks = np.array([len(r["t5"]) for r in m["records"]])
    Ds = np.array([r["dur"] for r in m["records"]])
    row("MEASURED", "保留的 clip 数", len(Ts),
        "cache_lj_uni/index.pt 的 records 长度（原始 13100 条按 1–10 秒过滤）")
    row("MEASURED", "说话人数", m["n_speakers"], "同上 n_speakers")
    row("MEASURED", "mel 归一化 mean", f"{m['latent_mean']:+.5f}",
        "preprocess_mel.py 在 400,000 帧随机样本上算的")
    row("MEASURED", "mel 归一化 std", f"{m['latent_std']:.5f}", "同上")
    row("MEASURED", "时长均值", f"{Ds.mean():.2f} 秒",
        "records 里 dur 的均值")
    row("MEASURED", "mel 帧数均值", f"{Ts.mean():.0f}", "records 里 T 的均值")
    row("MEASURED", "t5 token 均值 / p99",
        f"{Ks.mean():.1f} / {np.percentile(Ks, 99):.0f}",
        "records 里 len(t5) 的均值与 99 分位 —— max_text 48 就是据此选的")
    row("MEASURED", "短于 5.46 秒的比例", f"{100*(Ts <= 512).mean():.1f}%",
        "(T ≤ 512).mean()")
    row("MEASURED", "转录 ≤ 48 token 的比例", f"{100*(Ks <= 48).mean():.1f}%",
        "(len(t5) ≤ 48).mean()")

    # ---------------- MEASURED: the model
    from unified import UnifiedELF
    mod = UnifiedELF("ELF-B", max_length=560, whisper_dim=768)
    P = lambda x: sum(p.numel() for p in x.parameters())
    e = mod.elf
    row("MEASURED", "全模型参数", f"{P(mod):,}",
        "sum(p.numel() for p in UnifiedELF('ELF-B', 560, whisper_dim=768)"
        ".parameters())")
    row("MEASURED", "12 层 backbone 参数", f"{sum(P(b) for b in e.blocks):,}",
        "sum over elf.blocks")
    row("MEASURED", "backbone 占比",
        f"{sum(P(b) for b in e.blocks)/P(mod)*100:.1f}%", "上两行相除")
    row("MEASURED", "task_tokens 参数", f"{e.task_tokens.numel():,}",
        "elf.task_tokens.numel() = 2 任务 × 4 token × 768")
    row("MEASURED", "按任务分叉占比",
        f"{e.task_tokens.numel()/P(mod)*100:.4f}%", "上一行除以全模型")
    row("MEASURED", "voice_out 参数", f"{P(mod.voice_out):,}",
        "Linear(512, 100) 含 bias")
    row("MEASURED", "whisper_proj 参数", f"{P(mod.whisper_proj):,}",
        "Linear(768, 512) 含 bias")
    row("MEASURED", "self_cond_proj 形状",
        f"{tuple(e.self_cond_proj.weight.shape)}",
        "elf.self_cond_proj.weight.shape —— 1736 列就是拼接宽度")
    row("MEASURED", "unembed 形状", f"{tuple(e.unembed_kernel.shape)}",
        "elf.unembed_kernel.shape")
    row("MEASURED", "RoPE 表长度", e.feat_rope.freqs_cos.shape[0],
        "elf.feat_rope.freqs_cos.shape[0] = num_empty_token 16 + max_length 560"
        " = 576")

    # ---------------- MEASURED: a real batch's masks
    from torch.utils.data import DataLoader
    from train_unified import LibriCache, make_collate, S2T
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("t5-small")
    ds = LibriCache(os.path.join(HERE, "cache_lj_uni"), "test")
    itos = {v: k for k, v in m["stoi"].items()}
    coll = make_collate(560, 0.5, (0.0, 0.3), rng_seed=5, max_text=48,
                        itos=itos, tk=tk, layout="hybrid",
                        whisper_rate=m.get("whisper_rate", 0),
                        frame_rate=m["frame_rate"])
    b = next(iter(DataLoader(ds, batch_size=64, collate_fn=coll)))
    s = (b["task"] == S2T)
    for nm, mk in (("TTS", ~s), ("S2T", s)):
        va, ta = b["v_act"][mk], b["t_act"][mk]
        pv, pt = b["prompt_v"][mk], b["prompt_t"][mk]
        tgt = (ta * (1 - pt)) if nm == "S2T" else (va * (1 - pv))
        row("MEASURED", f"{nm} 的 flow 位置数", f"{tgt.sum(1).mean():.0f}",
            f"一个 64 行真实 batch：{'t_act·(1−prompt_t)' if nm=='S2T' else 'v_act·(1−prompt_v)'}"
            f" 的逐行求和再取均值（n={int(mk.sum())}）")

    # ---------------- MEASURED: the t distributions
    for pm, ps, label in ((-1.5, 0.8, "denoiser 的 t"),
                          (-0.5, 0.8, "decoder 的 λ（audio forcing 后）")):
        x = torch.sigmoid(torch.randn(400000) * ps + pm)
        row("MEASURED", f"{label} 的均值 / p95",
            f"{x.mean():.3f} / {np.percentile(x.numpy(), 95):.3f}",
            f"400,000 次 sigmoid(N({pm}, {ps})) 采样的均值与 95 分位")
        row("MEASURED", f"{label} 的 P(t>0.5)",
            f"{(x > 0.5).float().mean()*100:.1f}%", "同一组样本")

    # ---------------- print
    order = ["FIXED", "ELF", "OURS", "DERIVED", "MEASURED"]
    tag = {"FIXED": "外部固定，不能改",
           "ELF": "ELF 的配置，改就偏离预训练",
           "OURS": "我们选的，自由",
           "DERIVED": "由上面算出来的",
           "MEASURED": "从模型/batch/缓存打印的"}
    for k in order:
        rs = [r for r in ROWS if r[0] == k]
        print(f"\n{'='*78}\n{k}  —  {tag[k]}  （{len(rs)} 项）\n{'='*78}")
        for _, n, v, how in rs:
            print(f"  {n:<26} {v:>18}   {how}")
    print(f"\n共 {len(ROWS)} 个数字")


if __name__ == "__main__":
    main()
