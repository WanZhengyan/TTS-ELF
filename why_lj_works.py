#!/usr/bin/env python
"""Audio side by side, plus the measurements that separate the two corpora.

LJSpeech TTS works (gen WER 28-34, UTMOS 3.3) and LibriSpeech TTS does not
(163 clean). The two runs share the architecture, the objective and the
sampler, so the difference is in the data and in two numbers derived from it.
This script produces the audio to listen to and measures the candidates:

  * speakers, and clips PER speaker -- a voice the model must render from a
    3 s prompt is a different problem from one voice it can memorise;
  * how wrong the uniform text upsampling is on each corpus, using Whisper
    word timestamps as the reference alignment. On LibriSpeech that error was
    432 ms, 1.10 word durations. If LJSpeech is much lower, the uniform
    stretch is the explanation rather than a coincidence;
  * sequence length, since the exposure gap compounds along the trajectory.

For listening, the generated clip is emitted TWICE: the whole utterance, where
the first 3 s are the real recording that restore() pins (so the transition is
audible), and the generated span alone.
"""
import argparse
import base64
import io
import json
import os
import sys

import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from unified import Cfg, UnifiedELF
from train_unified import LibriCache, TextEncoder, make_collate
from sample_unified import tts_rollout


def log(*a):
    print(" ", *a, flush=True)


def wav_b64(x, sr):
    x = np.asarray(x, dtype=np.float32)
    pk = float(np.abs(x).max())
    if pk > 0:
        x = x / max(pk, 1e-6) * 0.97
    buf = io.BytesIO()
    torchaudio.save(buf, torch.from_numpy(x).unsqueeze(0), sr, format="wav")
    return base64.b64encode(buf.getvalue()).decode()


def align_error(recs, root, split, fr, asr, id2text, n, lj_root=None):
    """Mean |uniform start - true start| over words, and the word duration."""
    errs, wdur = [], []
    done = 0
    for r in recs:
        if done >= n:
            break
        uid = r["id"]
        if lj_root:
            p = os.path.join(lj_root, "wavs", uid + ".wav")
        else:
            if uid not in id2text:
                continue
            spk, chap, _ = uid.split("-")
            p = os.path.join(root, split, spk, chap, uid + ".flac")
        if not os.path.exists(p):
            continue
        w, sr = torchaudio.load(p)
        w = w.mean(0) if w.shape[0] > 1 else w[0]
        w16 = torchaudio.functional.resample(w, sr, 16000).numpy()
        out = asr(w16.copy(), return_timestamps="word",
                  generate_kwargs={"language": "en"})
        ch = [c for c in out.get("chunks", [])
              if c.get("timestamp") and c["timestamp"][0] is not None]
        if len(ch) < 3:
            continue
        dur = r["dur"]
        nw = len(ch)
        for k, c in enumerate(ch):
            errs.append(abs(dur * k / nw - float(c["timestamp"][0])))
        wdur.append(dur / nw)
        done += 1
    if not errs:
        return None
    e = np.array(errs)
    return dict(n_clips=done, n_words=len(errs), mean_ms=float(e.mean() * 1000),
                median_ms=float(np.median(e) * 1000),
                word_ms=float(np.mean(wdur) * 1000),
                ratio=float(e.mean() / np.mean(wdur)),
                frames=float(e.mean() * fr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libri_run", default="libri_tc100_uni_ddp")
    ap.add_argument("--lj_run", default="d_cond_fixed32")
    ap.add_argument("--libri_cache", default=os.path.join(HERE, "cache_libri_test_clean"))
    ap.add_argument("--lj_cache", default=os.path.join(HERE, "cache_lj_uni"))
    ap.add_argument("--libri_root", default=os.path.join(HERE, "data", "LibriSpeech"))
    ap.add_argument("--lj_root", default=os.path.join(HERE, "data", "LJSpeech-1.1"))
    ap.add_argument("--n_wav", type=int, default=4)
    ap.add_argument("--align_n", type=int, default=30)
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(HERE, "listen_why.json"))
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from vocos import Vocos
    voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(dev).eval()
    from transformers import AutoTokenizer, pipeline
    from transformers.models.whisper.english_normalizer import \
        EnglishTextNormalizer
    tk = AutoTokenizer.from_pretrained("t5-small")
    asr = pipeline("automatic-speech-recognition",
                   model="openai/whisper-large-v3", device=0,
                   torch_dtype=torch.float16)
    norm = EnglishTextNormalizer({})

    import glob
    id2text = {}
    for tr in glob.glob(os.path.join(args.libri_root, "test-clean",
                                     "*", "*", "*.trans.txt")):
        with open(tr, encoding="utf-8") as f:
            for line in f:
                uid, _, txt = line.strip().partition(" ")
                id2text[uid] = txt
    lj_text = {}
    mf = os.path.join(args.lj_root, "metadata.csv")
    if os.path.exists(mf):
        with open(mf, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 3:
                    lj_text[parts[0]] = parts[2]

    out = {"cases": [], "align": {}, "stats": {}}

    for tag, run, cache, is_lj, prompt_sec in (
            ("LJSpeech（能工作）", args.lj_run, args.lj_cache, True, 1.5),
            ("LibriSpeech（不工作）", args.libri_run, args.libri_cache, False, 3.0)):
        ck = torch.load(os.path.join(HERE, "runs", run, "ckpt.pt"),
                        map_location="cpu", weights_only=False)
        ta = ck["args"]
        dr = (4.0, 10.0) if not is_lj else None
        ds = LibriCache(cache, "all" if not is_lj else "test", dur_range=dr)
        meta = ds.meta
        fr = meta["frame_rate"]
        itos = {v: k for k, v in meta["stoi"].items()}
        wdim = meta.get("whisper_dim", 0) if ta.get("whisper") else 0
        wrate = meta.get("whisper_rate", 0.0) if ta.get("whisper") else 0.0
        model = UnifiedELF(ta["model"], char_vocab=meta["char_vocab_size"],
                           max_length=ta["max_length"], whisper_dim=wdim,
                           use_cond=not ta.get("no_cond", 0)).to(dev)
        model.load_state_dict(ck["ema"]); model.eval()
        te = TextEncoder(ta["text_latent_std"]).to(dev)
        cfg = Cfg(denoiser_p_mean=ta["denoiser_p_mean"])
        mel_stats = (meta["latent_mean"], meta["latent_std"])

        n_spk = meta["n_speakers"]
        n_clip = len(meta["records"])
        out["stats"][tag] = {
            "run": run, "speakers": n_spk, "clips": n_clip,
            "clips_per_speaker": n_clip / max(n_spk, 1),
            "hours": sum(r["dur"] for r in meta["records"]) / 3600,
            "hours_per_speaker": (sum(r["dur"] for r in meta["records"])
                                  / 3600 / max(n_spk, 1)),
            "max_length": ta["max_length"], "max_text": ta["max_text"],
            "mel_frames": ta["max_length"] - ta["max_text"],
            "mel_seconds": (ta["max_length"] - ta["max_text"]) / fr,
            "prompt_hi_trained": ta["prompt_hi"],
            "frames_per_t5": meta["frames_per_t5"],
        }
        log("")
        log(f"=== {tag}  ({run}) ===")
        for k, v in out["stats"][tag].items():
            log(f"    {k:22s} {v}")

        coll = make_collate(ta["max_length"], 0.0, (0.0, 0.0), rng_seed=7,
                            max_text=ta["max_text"], itos=itos, tk=tk,
                            layout=ta["layout"], whisper_rate=wrate,
                            frame_rate=fr, prompt_sec=prompt_sec)
        from torch.utils.data import DataLoader
        b = next(iter(DataLoader(ds, batch_size=args.n_wav, collate_fn=coll)))
        torch.manual_seed(0)
        mel = tts_rollout(model, b, cfg, dev, te, mel_stats, steps=args.nfe,
                          cfg_w=1.0)
        for i in range(min(args.n_wav, mel.shape[0])):
            uid = b["uid"][i]
            T = int(b["v_act"][i].sum()); K = int(b["n_prompt"][i])
            ref = lj_text.get(uid) if is_lj else id2text.get(uid)
            if ref is None:
                continue
            tm = torch.from_numpy(np.asarray(b["mel"][i, :T])).to(dev).unsqueeze(0)
            ceil = voc.decode(tm.transpose(1, 2))[0].cpu().numpy()
            full = voc.decode(mel[i:i + 1, :T].transpose(1, 2))[0].cpu().numpy()
            gen = voc.decode(mel[i:i + 1, K:T].transpose(1, 2))[0].cpu().numpy()
            g16 = torchaudio.functional.resample(
                torch.from_numpy(gen), 24000, 16000).numpy()
            hyp = norm(asr(g16.copy(), generate_kwargs={"language": "en"})["text"])
            out["cases"].append({
                "corpus": tag, "uid": uid, "ref": ref, "hyp_gen_only": hyp,
                "sec": T / fr, "prompt_sec": K / fr,
                "ceiling": wav_b64(ceil, 24000),
                "full": wav_b64(full, 24000),
                "gen_only": wav_b64(gen, 24000)})
            log(f"    {uid}  {T/fr:.2f}s  prompt {K/fr:.2f}s")
            log(f"      参考      {ref[:90]}")
            log(f"      仅生成区  {hyp[:90]}")
        del model, te
        torch.cuda.empty_cache()

        # alignment error on this corpus
        a = align_error(meta["records"], args.libri_root, "test-clean", fr, asr,
                        id2text if not is_lj else lj_text, args.align_n,
                        lj_root=args.lj_root if is_lj else None)
        out["align"][tag] = a
        if a:
            log(f"    均匀拉伸的对齐误差: 均值 {a['mean_ms']:.0f} ms "
                f"= {a['ratio']:.2f} 个词长 = {a['frames']:.0f} 帧 "
                f"(词长 {a['word_ms']:.0f} ms, {a['n_words']} 词)")

    json.dump(out, open(args.out, "w"))
    log("")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
