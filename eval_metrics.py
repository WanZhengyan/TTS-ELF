#!/usr/bin/env python
"""The metrics papers actually report: WER and UTMOS.

Runs on four tracks so every number has its references attached:

  gt        the original waveform. Its WER is NOT zero -- it is the ASR's own floor,
            and every other row must be read against it.
  ceiling   original -> codec encode -> VQ -> decode. The upper bound on anything
            this system can produce. The row most papers forget and reviewers ask for.
  gen       full ODE rollout + CE decode. The system.
  gen_tf    true x0 handed to the CE head. Isolates the discretizer from the flow.

WER uses Whisper-large-v3 with Whisper's own EnglishTextNormalizer, which is the
Seed-TTS-eval convention. Note LibriSpeech-PC papers use a HuBERT-based ASR instead,
so numbers are only comparable within one ASR choice.
"""
import argparse, json, os, sys

import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "third_party", "WavTokenizer"))
sys.path.insert(0, HERE)
from train import ELFTTS, LJCache, make_collate, Cfg, sample as elf_sample  # noqa: E402
from decoder.pretrained import WavTokenizer  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def log(*a):
    print(*a, flush=True)


def wer_pair(ref, hyp):
    """Word error rate by Levenshtein on word tokens."""
    r, h = ref.split(), hyp.split()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r), len(h)], len(r)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "test2", "ckpt.pt"))
    ap.add_argument("--cache", default=os.path.join(HERE, "cache"))
    ap.add_argument("--lj_root", default=os.path.join(HERE, "data", "LJSpeech-1.1"))
    ap.add_argument("--codec_cfg", default=os.path.join(
        HERE, "third_party", "WavTokenizer", "configs",
        "wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"))
    ap.add_argument("--codec_ckpt", default=os.path.join(
        HERE, "ckpt", "wavtokenizer_large_unify_600_24k.ckpt"))
    ap.add_argument("--weights", default="ema")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--n_steps", type=int, default=32)
    ap.add_argument("--cfg_scale", type=float, default=2.0)
    ap.add_argument("--prompt_frac", type=float, default=0.0,
                    help="0.0 = pure TTS, the LJSpeech standard")
    ap.add_argument("--asr", default="openai/whisper-large-v3")
    ap.add_argument("--skip_model", action="store_true",
                    help="only measure gt and ceiling -- no checkpoint needed")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "metrics.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- ASR
    from transformers import pipeline
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    log(f"loading ASR {args.asr} ...")
    asr = pipeline("automatic-speech-recognition", model=args.asr,
                   device=0 if device.type == "cuda" else -1,
                   torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    norm = EnglishTextNormalizer({})

    # ---- UTMOS
    log("loading UTMOS ...")
    try:
        utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                               trust_repo=True).to(device).eval()
    except Exception as e:
        log(f"  UTMOS unavailable ({type(e).__name__}: {e}); continuing without it")
        utmos = None

    # ---- codec
    wt = WavTokenizer.from_pretrained0802(args.codec_cfg, args.codec_ckpt).to(device).eval()
    vq = wt.feature_extractor.encodec.quantizer.vq.layers[0]
    bw = torch.tensor([0], device=device)

    # ---- data
    ds = LJCache(args.cache, "test", seed=0)
    meta = ds.meta
    id2text = {}
    with open(os.path.join(args.lj_root, "metadata.csv"), encoding="utf-8") as f:
        for line in f:
            p = line.strip().split("|")
            if len(p) >= 3:
                id2text[p[0]] = p[2]

    model = None
    ta = {"model": "-", "steps": 0, "max_length": 400, "seed": 0}
    if not args.skip_model:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        ta = ck["args"]
        cfg = Cfg(**{k: v for k, v in ck["cfg"].items() if k in Cfg.__dataclass_fields__})
        model = ELFTTS(ta["model"], meta["char_vocab_size"], meta["audio_vocab_size"],
                       dim=512, max_length=ta["max_length"],
                       cb_skip=ta.get("cb_skip", False)).to(device)
        model.load_state_dict(ck[args.weights]); model.eval()
        log(f"model {ta['model']} @ {ta['steps']} steps ({args.weights})")
    else:
        cfg = Cfg()
        log("skip_model: measuring gt + ceiling only")

    coll = make_collate(ta["max_length"], (args.prompt_frac, args.prompt_frac),
                        ta["seed"], fixed_prompt=args.prompt_frac)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=coll)
    resample = torchaudio.transforms.Resample(22050, 24000).to(device)

    tracks = ["gt", "ceiling"] + ([] if args.skip_model else ["gen", "gen_tf"])
    errs = {t: 0 for t in tracks}
    lens = {t: 0 for t in tracks}
    mos = {t: [] for t in tracks}
    code_hit = code_tot = 0.0
    done = 0

    for bi, b in enumerate(dl):
        if done >= args.n:
            break
        x0 = b["latent"].to(device) / cfg.latent_std
        code_true = b["code"].to(device)
        text_ids = b["text"].to(device)
        m = b["valid"].to(device) * (1 - b["prompt"].to(device))
        B = x0.shape[0]

        preds = {}
        if model is not None:
            p, _, _ = elf_sample(model, b, cfg, device, n_steps=args.n_steps,
                                 cfg_scale=args.cfg_scale, schedule="logit_normal")
            preds["gen"] = p
            t1 = torch.ones(B, device=device); sc = torch.ones(B, device=device)
            _, lg = model(x0, torch.zeros_like(x0), text_ids, t1,
                          self_cond_cfg_scale=sc, decoder_step_active=True)
            preds["gen_tf"] = lg.argmax(-1)
            code_hit += ((p == code_true).float() * m).sum().item()
            code_tot += m.sum().item()

        for i in range(B):
            if done >= args.n:
                break
            uid = b["id"][i]; T = int(b["T"][i])
            ref = norm(id2text.get(uid, ""))
            if not ref:
                continue

            waves = {}
            w, _ = torchaudio.load(os.path.join(args.lj_root, "wavs", uid + ".wav"))
            g = resample(w.to(device))
            waves["gt"] = g.mean(0, keepdim=True) if g.shape[0] > 1 else g
            waves["ceiling"] = wt.decode(vq.decode(code_true[i:i + 1, :T]), bandwidth_id=bw)
            for k, p in preds.items():
                waves[k] = wt.decode(vq.decode(p[i:i + 1, :T]), bandwidth_id=bw)

            for t in tracks:
                a = waves[t].squeeze(0).float().clamp(-1, 1)
                out = asr({"sampling_rate": 24000, "raw": a.cpu().numpy()},
                          generate_kwargs={"language": "english"})
                e, n = wer_pair(ref, norm(out["text"]))
                errs[t] += e; lens[t] += n
                if utmos is not None:
                    mos[t].append(float(utmos(a.unsqueeze(0), 24000).item()))
            done += 1
            if done % 10 == 0:
                log(f"  {done}/{args.n}  " + "  ".join(
                    f"{t} WER {100*errs[t]/max(lens[t],1):.2f}" for t in tracks))

    log("")
    log(f"=== {done} held-out clips, prompt {args.prompt_frac:.0%}"
        f"{' (pure TTS)' if args.prompt_frac == 0 else ''} ===")
    log(f"{'track':<10} {'WER %':>8} {'UTMOS':>8}")
    res = {"n": done, "prompt_frac": args.prompt_frac, "asr": args.asr,
           "model": ta["model"], "steps": ta["steps"], "tracks": {}}
    for t in tracks:
        w = 100 * errs[t] / max(lens[t], 1)
        u = float(np.mean(mos[t])) if mos[t] else float("nan")
        log(f"{t:<10} {w:>8.2f} {u:>8.3f}")
        res["tracks"][t] = {"wer": w, "utmos": u, "n_words": lens[t]}
    if code_tot:
        res["code_acc"] = code_hit / code_tot
        log(f"\ncode accuracy (generated): {100*code_hit/code_tot:.2f}%")
    log("")
    log("read it this way: `gt` is the ASR's floor, not zero. `ceiling` is what the codec")
    log("alone costs you. `gen` minus `ceiling` is the generative model's own gap.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
