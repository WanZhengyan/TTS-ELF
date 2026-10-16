#!/usr/bin/env python
"""Listening examples for the report from one checkpoint of the two-source run.

Takes the FIRST n pairs of the LJSpeech eval list (no picking), builds each
input exactly as eval_lj.py does (prompt mel + "prompt text target text",
F5's estimated target length), samples with CFG w, and stores base64 WAVs of
the prompt, the real target recording and the generated span, plus a
Whisper-large-v3 transcript of the generated span. Runs on CPU if it must.
"""
import argparse
import base64
import io
import json
import os
import pickle
import sys

import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from unified import TTS, Cfg
from train_unified import TextEncoder, make_collate
from sample_unified import tts_rollout
from eval_lj import load_model, f5_norm, utt_wer


def b64wav(w16):
    buf = io.BytesIO()
    torchaudio.save(buf, torch.from_numpy(np.asarray(w16, np.float32)).unsqueeze(0),
                    16000, format="wav", encoding="PCM_S", bits_per_sample=16)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--list", default=os.path.join(HERE, "data", "lj_eval", "lj_inset_1000.lst"))
    ap.add_argument("--lj_cache", default=os.path.join(HERE, "cache_lj13k"))
    ap.add_argument("--lj_root", default=os.path.join(HERE, "data", "LJSpeech-1.1"))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ta = ck["args"]
    model, meta = load_model(ck, dev)
    mk = ck["meta_keys"]
    mel_stats = (mk["latent_mean"], mk["latent_std"])
    cfg = Cfg(denoiser_p_mean=ta["denoiser_p_mean"], decoder_p_mean=ta["decoder_p_mean"])
    te = TextEncoder(ta["text_latent_std"]).to(dev)
    from transformers import AutoTokenizer, pipeline
    tk = AutoTokenizer.from_pretrained("t5-small")
    stoi = meta["stoi"]
    lj = pickle.load(open(os.path.join(a.lj_cache, "index.pt"), "rb"))
    by_id = {r["id"]: r for r in lj["records"]}
    lat = np.memmap(os.path.join(a.lj_cache, "latents.f16"), dtype=np.float16, mode="r",
                    shape=(lj["total_frames"], lj["dim"]))
    rows = [l.rstrip("\n").split("\t") for l in open(a.list, encoding="utf-8")][:a.n]
    fr = meta["frame_rate"]
    items, KT = [], []
    for pu, _, pt, tu, _, tt in rows:
        pr = by_id[pu]
        pm = np.asarray(lat[pr["offset"]:pr["offset"] + pr["T"]], dtype=np.float32)
        K = pm.shape[0]
        Tg = int(K / len(pt.encode("utf-8")) * len(tt.encode("utf-8")))
        Tg = min(Tg, ta["max_length"] - ta["max_text"] - K)
        sep = " " if len(pt[-1].encode("utf-8")) == 1 else ""
        full = (pt + sep + tt).lower().strip()
        items.append({"mel": np.concatenate([pm, np.zeros((Tg, pm.shape[1]), np.float32)]),
                      "chars": np.array([stoi.get(c, 1) for c in full], dtype=np.int16),
                      "t5": np.asarray(tk(full).input_ids, dtype=np.int32),
                      "spk": 0, "uid": tu, "task": TTS})
        KT.append((K, Tg))
    coll = make_collate(ta["max_length"], 0.0, (0.0, 0.0), rng_seed=7,
                        max_text=ta["max_text"], itos={v: k for k, v in stoi.items()},
                        tk=tk, layout=ta["layout"], whisper_rate=0.0, frame_rate=fr)
    b = coll(items)
    b["prompt_v"].zero_()
    for i, (K, _) in enumerate(KT):
        b["prompt_v"][i, :K] = 1.0
        b["n_prompt"][i] = K
    torch.manual_seed(a.seed)
    with torch.no_grad():
        mel = tts_rollout(model, b, cfg, dev, te, mel_stats, steps=a.nfe, cfg_w=a.cfg)
    from vocos import Vocos
    voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(dev).eval()
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3",
                   device=0 if dev.type == "cuda" else -1,
                   dtype=torch.float16 if dev.type == "cuda" else torch.float32)

    def wav16(path):
        w, sr = torchaudio.load(path)
        w = w.mean(0) if w.shape[0] > 1 else w[0]
        return torchaudio.functional.resample(w, sr, 16000).numpy()

    out = {"ckpt": a.ckpt, "step": ck["step"], "nfe": a.nfe, "cfg": a.cfg,
           "seed": a.seed, "cases": []}
    for i, (pu, pd, pt, tu, td, tt) in enumerate(rows):
        K, Tg = KT[i]
        with torch.no_grad():
            g = voc.decode(mel[i:i + 1, K:K + Tg].transpose(1, 2))[0].cpu()
        g16 = torchaudio.functional.resample(g, 24000, 16000).numpy()
        hyp = asr(g16.copy(), generate_kwargs={"language": "en", "num_beams": 5})["text"]
        w, c = utt_wer(f5_norm(tt), f5_norm(hyp))
        out["cases"].append({
            "prompt_uid": pu, "prompt_sec": float(pd), "prompt_text": pt,
            "target_uid": tu, "target_sec": float(td), "target_text": tt,
            "gen_sec": Tg / fr, "hyp": hyp.strip(), "wer": 100 * w,
            "sub": c["sub"], "dele": c["dele"], "ins": c["ins"],
            "prompt_wav": b64wav(wav16(os.path.join(a.lj_root, "wavs", pu + ".wav"))),
            "target_wav": b64wav(wav16(os.path.join(a.lj_root, "wavs", tu + ".wav"))),
            "gen_wav": b64wav(g16)})
        print(f"{tu}: WER {100*w:.1f}  ref: {tt}\n          hyp: {hyp.strip()}", flush=True)
    json.dump(out, open(a.out, "w"))
    print(f"wrote {a.out} (step {ck['step']})")


if __name__ == "__main__":
    main()
