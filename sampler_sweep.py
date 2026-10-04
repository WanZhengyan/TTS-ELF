#!/usr/bin/env python
"""ODE vs SDE on the existing checkpoint — no retraining.

ELF's own unconditional config samples with SDE at gamma=1.5, and ELF-S2T uses
sde gamma=1.5 as well. I had taken ODE from ELF's *conditional* config. An SDE
step churns z back toward noise before each prediction, which re-anchors the
trajectory on the manifold the model was trained on instead of letting
integration error accumulate -- which is exactly the failure measured earlier
(exposure gap 15x at t=0.7).
"""
import os, sys, json, torch, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,"third_party","WavTokenizer")); sys.path.insert(0,HERE)
from train import ELFTTS, LJCache, make_collate, Cfg, load_codebook, sample as S
from decoder.pretrained import WavTokenizer
from torch.utils.data import DataLoader

dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck=torch.load(f"{HERE}/runs/d_xs_dp02/ckpt.pt",map_location="cpu",weights_only=False)
ta=ck["args"]; cfg=Cfg(**{k:v for k,v in ck["cfg"].items() if k in Cfg.__dataclass_fields__})
val=LJCache(f"{HERE}/cache","val",seed=ta["seed"]); meta=val.meta
m_=ELFTTS(ta["model"],meta["char_vocab_size"],meta["audio_vocab_size"],dim=512,
          max_length=ta["max_length"]).to(dev); m_.load_state_dict(ck["ema"]); m_.eval()
coll=make_collate(ta["max_length"],(0.0,0.0),ta["seed"],fixed_prompt=0.0)
b=next(iter(DataLoader(val,batch_size=16,shuffle=False,num_workers=0,collate_fn=coll)))
wt=WavTokenizer.from_pretrained0802(
    f"{HERE}/third_party/WavTokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
    f"{HERE}/ckpt/wavtokenizer_large_unify_600_24k.ckpt").to(dev).eval()
vq=wt.feature_extractor.encodec.quantizer.vq.layers[0]; bw=torch.tensor([0],device=dev)
try:
    um=torch.hub.load("tarepan/SpeechMOS:v1.2.0","utmos22_strong",trust_repo=True).to(dev).eval()
except Exception as e:
    print("no UTMOS:",e); um=None

x0=b["latent"].to(dev)/cfg.latent_std; code=b["code"].to(dev)
msk=b["valid"].to(dev)*(1-b["prompt"].to(dev))
nrm=lambda a:((a.norm(dim=-1)*msk).sum()/msk.sum().clamp(min=1)).item()

def ent(c):
    f=c[msk>0].flatten(); h=torch.bincount(f,minlength=4096).float(); p=h/h.sum(); p=p[p>0]
    return float(-(p*p.log2()).sum()), int((h>0).sum())

ref=[wt.decode(vq.decode(code[i:i+1,:int(b["T"][i])]),bandwidth_id=bw) for i in range(4)]
mos_ref=float(np.mean([um(w.squeeze(0).float().clamp(-1,1).unsqueeze(0),24000).item() for w in ref])) if um else float("nan")
e_ref,u_ref=ent(code)

print(f"ckpt {ta['model']} @ {ta['steps']} steps | pure TTS (prompt=0) | 16 clips")
print(f"{'sampler':>7} {'gamma':>6} {'steps':>6} {'cfg':>5} | {'||z-x0||':>9} {'UTMOS':>7} {'bits':>6} {'uniq':>5} {'code acc':>9}")
print("-"*82)
rows=[]
for sampler,gammas in (("ode",[0.0]),("sde",[0.5,1.0,1.5,3.0])):
    for g in gammas:
        for ns in (32,64):
            torch.manual_seed(0)
            pred,z,nfe=S(m_,b,cfg,dev,n_steps=ns,cfg_scale=2.0,schedule="logit_normal",
                         sampler=sampler,sde_gamma=g)
            acc=(((pred==code).float()*msk).sum()/msk.sum()).item()
            e,u=ent(pred)
            wavs=[wt.decode(vq.decode(pred[i:i+1,:int(b["T"][i])]),bandwidth_id=bw) for i in range(4)]
            mos=float(np.mean([um(w.squeeze(0).float().clamp(-1,1).unsqueeze(0),24000).item() for w in wavs])) if um else float("nan")
            print(f"{sampler:>7} {g:>6.1f} {ns:>6} {2.0:>5.1f} | {nrm(z-x0):>9.3f} {mos:>7.3f} {e:>6.2f} {u:>5} {acc*100:>8.2f}%")
            rows.append(dict(sampler=sampler,gamma=g,steps=ns,z_err=nrm(z-x0),utmos=mos,
                             bits=e,uniq=u,code_acc=acc))
print(f"{'ceiling':>7} {'-':>6} {'-':>6} {'-':>5} | {0.0:>9.3f} {mos_ref:>7.3f} {e_ref:>6.2f} {u_ref:>5} {100.0:>8.2f}%")
json.dump(rows,open(f"{HERE}/out/sampler_sweep.json","w"),indent=2)
print("\nODE was my choice, taken from ELF's CONDITIONAL config. ELF's own")
print("unconditional config and ELF-S2T both use SDE at gamma=1.5.")
