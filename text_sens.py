#!/usr/bin/env python
"""Is the model using the text channel at all?

CFG scale barely moved the rollout (26.43 -> 26.39), which is suspicious: if the
conditional and unconditional velocity fields are nearly identical, the model has
learned to ignore text, and then sampled code accuracy is at chance BY CONSTRUCTION
regardless of how good the flow is.

Measures, at several t: ||x0_hat(text) - x0_hat(no text)|| relative to ||x0_hat||,
and the same for real text vs SHUFFLED text (a control: shuffling should hurt as
much as zeroing if the content matters).
"""
import os, sys, json
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from train import ELFTTS, LJCache, make_collate, Cfg, restore
from torch.utils.data import DataLoader

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(f"{HERE}/runs/test2/ckpt.pt", map_location="cpu", weights_only=False)
ta = ck["args"]; cfg = Cfg(**{k: v for k, v in ck["cfg"].items() if k in Cfg.__dataclass_fields__})
val = LJCache(f"{HERE}/cache", "val", seed=ta["seed"]); meta = val.meta
coll = make_collate(ta["max_length"], (0.3, 0.3), ta["seed"], fixed_prompt=0.3)
dl = DataLoader(val, batch_size=16, shuffle=False, num_workers=0, collate_fn=coll)
m_ = ELFTTS(ta["model"], meta["char_vocab_size"], meta["audio_vocab_size"], dim=512,
            max_length=ta["max_length"]).to(dev)
m_.load_state_dict(ck["ema"]); m_.eval()

print(f"ELF-XS @ {ta['steps']} steps, ema | label_drop_prob was {cfg.label_drop_prob}")
print(f"{'t':>6} {'||xh||':>8} {'||d text-vs-zero||':>19} {'rel':>7} {'||d text-vs-shuf||':>19} {'rel':>7}")
out=[]
with torch.no_grad():
    for tv in (0.05,0.1,0.2,0.3,0.5,0.7,0.85):
        acc_n=acc_z=acc_s=0.0; nb=0
        for bi,b in enumerate(dl):
            if bi>=4: break
            x0=b["latent"].to(dev)/cfg.latent_std
            text=b["text"].to(dev); prompt=b["prompt"].to(dev)
            msk=b["valid"].to(dev)*(1-prompt)
            B=x0.shape[0]
            t=torch.full((B,),tv,device=dev)
            torch.manual_seed(bi)  # same noise across the three conditions
            eps=torch.randn_like(x0)
            z=restore(t.view(-1,1,1)*x0+(1-t.view(-1,1,1))*eps*cfg.denoiser_noise_scale,x0,prompt)
            sc=torch.ones(B,device=dev); zsc=restore(torch.zeros_like(z),x0,prompt)
            xh  = m_(z,zsc,text,t,self_cond_cfg_scale=sc,drop_text=torch.zeros(B,device=dev))[0]
            xh0 = m_(z,zsc,text,t,self_cond_cfg_scale=sc,drop_text=torch.ones(B,device=dev))[0]
            # control: shuffle characters along time within each example
            perm=torch.stack([text[i][torch.randperm(text.shape[1],device=dev)] for i in range(B)])
            xhs = m_(z,zsc,perm,t,self_cond_cfg_scale=sc,drop_text=torch.zeros(B,device=dev))[0]
            n=lambda a: ((a.norm(dim=-1)*msk).sum()/msk.sum().clamp(min=1)).item()
            acc_n+=n(xh); acc_z+=n(xh-xh0); acc_s+=n(xh-xhs); nb+=1
        a,bz,bs=acc_n/nb,acc_z/nb,acc_s/nb
        print(f"{tv:>6.2f} {a:>8.3f} {bz:>19.4f} {bz/a*100:>6.2f}% {bs:>19.4f} {bs/a*100:>6.2f}%")
        out.append({"t":tv,"xh_norm":a,"d_zero":bz,"d_shuf":bs})
json.dump(out,open(f"{HERE}/out/text_sens.json","w"),indent=2)
print("\nIf both relative columns are ~0%, the text channel is being ignored.")
