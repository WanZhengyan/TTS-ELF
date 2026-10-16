#!/usr/bin/env python
"""Cash in the unconditional branch that audio_drop_prob trained but nobody used.

`audio_drop_prob` zeroes the Whisper features AND the mel target on a fraction of
the S2T rows, which is how ELF-S2T builds an unconditional branch for audio
classifier-free guidance (ACFG, w=2.0 in that paper). Our trainer has had that
dropout since p_dp5_acfg, but no sampler here ever ran the unconditional pass, so
the branch was pure training cost with no payoff. This script runs it at
inference only -- no retraining -- over a grid of w:

    x0_hat = x0_uncond + w * (x0_cond - x0_uncond)        w=1 is plain sampling

A checkpoint trained WITHOUT the dropout has no valid unconditional branch, so
guidance there should make things worse. That is the falsification check, and it
is why this sweep always reports such a control next to the real one.
"""
import argparse, json, os, sys

import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from unified import Cfg, UnifiedELF, net_to_v, restore
from train_unified import (LibriCache, TextEncoder, make_collate, build_targets,
                           make_schedule, collapse, wer_pair, audio_cond,
                           T5_EOS, T5_PAD)


def log(*a):
    print(" ", *a, flush=True)


@torch.no_grad()
def s2t_acfg(model, b, cfg, device, text_enc, mel_stats, tk, layout, steps, w):
    """One S2T rollout. w == 1.0 takes the single-pass path, so the w=1 row of
    the sweep is bit-for-bit the ordinary sampler and not a guided special case."""
    chars = b["chars"].to(device); valid = b["valid"].to(device)
    v_act = b["v_act"].to(device).unsqueeze(-1)
    t_act = b["t_act"].to(device).unsqueeze(-1)
    p_v, p_t = b["prompt_v"].to(device), b["prompt_t"].to(device)
    task = b["task"].to(device)
    x0_t, x0_v = build_targets(b, text_enc, device, mel_stats)
    ac, am = audio_cond(b, device)
    B = x0_t.shape[0]
    ns = cfg.denoiser_noise_scale

    # the unconditional branch is exactly what training produced on a dropped
    # row: Whisper features zeroed and the mel channel zeroed together
    ac_u = torch.zeros_like(ac) if ac is not None else None
    guided = (w != 1.0)

    z_t = restore(torch.randn_like(x0_t) * ns, x0_t, p_t) * t_act
    z_v = restore(torch.randn_like(x0_v) * ns, x0_v, p_v) * v_act
    xp_t, xp_v = torch.zeros_like(z_t), torch.zeros_like(z_v)
    sc_cfg = torch.ones(B, device=device)
    ts = make_schedule(steps, cfg, device)

    for i in range(steps):
        tv = float(ts[i])
        t = torch.full((B,), tv, device=device)
        xh_t, xh_v, _ = model(z_t, z_v, xp_t, xp_v, chars, t, task,
                              attention_mask=valid, self_cond_cfg_scale=sc_cfg,
                              audio_cond=ac, audio_cond_mask=am)
        if guided:
            zv_u = torch.zeros_like(z_v)
            u_t, _, _ = model(z_t, zv_u, xp_t, torch.zeros_like(xp_v), chars, t,
                              task, attention_mask=valid,
                              self_cond_cfg_scale=sc_cfg,
                              audio_cond=ac_u, audio_cond_mask=am)
            # guidance is applied on the text channel only: the mel channel is
            # the conditioning side here, not something we are generating
            xh_t = u_t + w * (xh_t - u_t)
        h = float(ts[i + 1]) - tv
        z_t = restore(z_t + h * net_to_v(xh_t, z_t, t), x0_t, p_t) * t_act
        z_v = restore(z_v + h * net_to_v(xh_v, z_v, t), x0_v, p_v) * v_act
        xp_t = restore(xh_t, x0_t, p_t) * t_act
        xp_v = restore(xh_v, x0_v, p_v) * v_act

    t1 = torch.ones(B, device=device)
    _, _, logits = model(z_t, z_v, xp_t, xp_v, chars, t1, task,
                         attention_mask=valid, self_cond_cfg_scale=sc_cfg,
                         decoder_step_active=True,
                         audio_cond=ac, audio_cond_mask=am)
    pred = logits.argmax(-1)
    err = ln = tok = tokn = 0
    pairs = []
    for i in range(B):
        n = int(b["n_text"][i]); st = int(b["text_start"][i])
        ref_ids = [x for x in b["t5_dense"][i, :n].tolist()
                   if x not in (T5_EOS, T5_PAD)]
        if layout == "channel":
            hyp_ids = collapse(pred[i].tolist())
        else:
            raw = pred[i, st:st + b["t5_dense"].shape[1]].tolist()
            if T5_EOS in raw:
                raw = raw[:raw.index(T5_EOS)]
            hyp_ids = [x for x in raw if x != T5_PAD]
        ref = tk.decode(ref_ids, skip_special_tokens=True)
        hyp = tk.decode(hyp_ids, skip_special_tokens=True)
        e, l = wer_pair(ref, hyp)
        err += e; ln += l
        k = min(len(ref_ids), len(hyp_ids))
        tok += sum(1 for j in range(k) if ref_ids[j] == hyp_ids[j])
        tokn += len(ref_ids)
        pairs.append((ref, hyp))
    return (err, ln, tok, tokn), pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="p_dp5_acfg,p_dp5")
    ap.add_argument("--cache", default=os.path.join(HERE, "cache_lj_uni"))
    ap.add_argument("--w", default="1.0,1.5,2.0,3.0")
    ap.add_argument("--n", type=int, default=288)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--split", default="test",
                    help="'all' 用完整官方测试集（文献报的是这个）；"
                         "'test' 只切 300 条")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "acfg_sweep.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("t5-small")
    ws = [float(x) for x in args.w.split(",")]
    results = {}

    for run in args.runs.split(","):
        ckpt = os.path.join(HERE, "runs", run, "ckpt.pt")
        if not os.path.exists(ckpt):
            log(f"skip {run}: no checkpoint"); continue
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        ta = ck["args"]
        trained_drop = float(ta.get("audio_drop_prob") or 0.0)
        ds = LibriCache(args.cache, args.split)
        meta = ds.meta
        mel_stats = (meta["latent_mean"], meta["latent_std"])
        itos = {v: k for k, v in meta["stoi"].items()}
        cfg = Cfg(denoiser_p_mean=ta["denoiser_p_mean"])
        wdim = meta.get("whisper_dim", 0) if ta.get("whisper") else 0
        wrate = meta.get("whisper_rate", 0.0) if ta.get("whisper") else 0.0
        model = UnifiedELF(ta["model"], char_vocab=meta["char_vocab_size"],
                           max_length=ta["max_length"], whisper_dim=wdim,
                           use_cond=not ta.get("no_cond", 0)).to(device)
        model.load_state_dict(ck["ema"]); model.eval()
        text_enc = TextEncoder(ta["text_latent_std"]).to(device)
        coll = make_collate(ta["max_length"], 1.0, (0.0, 0.0), rng_seed=7,
                            max_text=ta["max_text"], itos=itos, tk=tk,
                            layout=ta["layout"], whisper_rate=wrate,
                            frame_rate=meta["frame_rate"])

        log("")
        log(f"=== {run}  step {ck['step']}  "
            f"trained audio_drop_prob {trained_drop} "
            f"({'has' if trained_drop > 0 else 'NO'} unconditional branch) ===")
        results[run] = {"trained_drop": trained_drop, "step": ck["step"], "w": {}}
        for w in ws:
            torch.manual_seed(7)
            dl = DataLoader(ds, batch_size=args.batch, collate_fn=coll)
            E = [0, 0, 0, 0]; seen = 0; shown = []
            for b in dl:
                if seen >= args.n:
                    break
                c, pr = s2t_acfg(model, b, cfg, device, text_enc, mel_stats, tk,
                                 ta["layout"], args.nfe, w)
                for k in range(4):
                    E[k] += c[k]
                seen += len(pr)
                if not shown:
                    shown = pr[:3]
            wer = 100.0 * E[0] / max(E[1], 1)
            acc = E[2] / max(E[3], 1)
            results[run]["w"][str(w)] = {"wer": wer, "tok_acc": acc, "n": seen}
            log(f"  w={w:<4}  WER {wer:6.2f}%   token acc {acc*100:5.2f}%   "
                f"(n={seen} clips)")
            for ref, hyp in shown:
                log(f"      ref: {ref[:70]}")
                log(f"      hyp: {hyp[:70]}")
        del model, text_enc
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    log("")
    log(f"wrote {args.out}")
    # the verdict is a comparison, not a single number: guidance must help the
    # dropout-trained model and must NOT help the control
    for run, r in results.items():
        base = r["w"].get("1.0", {}).get("wer")
        if base is None:
            continue
        best_w, best = min(((k, v["wer"]) for k, v in r["w"].items()),
                           key=lambda kv: kv[1])
        log(f"{run:12s} drop={r['trained_drop']}  w=1 {base:6.2f}%  "
            f"best w={best_w} {best:6.2f}%  delta {best - base:+.2f}")


if __name__ == "__main__":
    main()
