#!/usr/bin/env python
"""How much of the eval-to-eval movement in z_err_ratio is just sampling noise?

The ratio fell 1.116 -> 1.104 -> 1.087 -> 1.074 and then rose to 1.079. I called
that rise "within noise" without having measured the noise. The rollout draws
fresh Gaussian noise each time, so the same checkpoint on the same batch gives a
distribution, not a number. This measures its spread so the trend can be read
against a real error bar.
"""
import os, sys, json, numpy as np, torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from train import ELFTTS, LJCache, make_collate, Cfg, get_sampling_steps, restore, net_to_v
from torch.utils.data import DataLoader

ap=__import__("argparse").ArgumentParser()
ap.add_argument("--ckpt", default=f"{HERE}/runs/h_mel_xs/ckpt.pt")
ap.add_argument("--cache", default=f"{HERE}/cache_mel")
ap.add_argument("--reps", type=int, default=12)
ap.add_argument("--weights", default="raw")
a=ap.parse_args()

dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck=torch.load(a.ckpt,map_location="cpu",weights_only=False); ta=ck["args"]
cfg=Cfg(**{k:v for k,v in ck["cfg"].items() if k in Cfg.__dataclass_fields__})
ds=LJCache(a.cache,"val",seed=ta["seed"]); meta=ds.meta
coll=make_collate(ta["max_length"],(0.0,0.0),ta["seed"],fixed_prompt=0.0,dim=meta["dim"])
b=next(iter(DataLoader(ds,batch_size=16,shuffle=False,num_workers=0,collate_fn=coll)))
m_=ELFTTS(ta["model"],meta["char_vocab_size"],meta["audio_vocab_size"],dim=meta["dim"],
          max_length=ta["max_length"]).to(dev)
m_.load_state_dict(ck["model" if a.weights=="raw" else "ema"]); m_.eval()

x0=(b["latent"].to(dev)-cfg.latent_mean)/cfg.latent_std
text=b["text"].to(dev); prompt=b["prompt"].to(dev)
msk=b["valid"].to(dev)*(1-prompt)
nrm=lambda z:((z.norm(dim=-1)*msk).sum()/msk.sum().clamp(min=1)).item()
triv=nrm(x0)

@torch.no_grad()
def once(seed):
    torch.manual_seed(seed)
    z=restore(torch.randn_like(x0)*cfg.denoiser_noise_scale,x0,prompt)
    xp=restore(torch.zeros_like(z),x0,prompt)
    ts=get_sampling_steps(32,"logit_normal",cfg.denoiser_p_mean,cfg.denoiser_p_std,dev)
    B=x0.shape[0]; scb=torch.ones(B,device=dev)
    zd,od=torch.zeros(B,device=dev),torch.ones(B,device=dev)
    for i in range(len(ts)-1):
        h=float(ts[i+1]-ts[i]); al=max(0.,min(1.,1.-1.0*h))
        z=restore(al*z+(1-al)*torch.randn_like(z)*cfg.denoiser_noise_scale,x0,prompt)
        tc=al*float(ts[i]); t=torch.full((B,),tc,device=dev)
        v_c,x_c=net_to_v(m_(z,xp,text,t,self_cond_cfg_scale=scb,drop_text=zd),z,t)
        v_c=restore(v_c,torch.zeros_like(z),prompt); x_c=restore(x_c,x0,prompt)
        v_u,_=net_to_v(m_(z,xp,text,t,self_cond_cfg_scale=scb,drop_text=od),z,t)
        v_u=restore(v_u,torch.zeros_like(z),prompt)
        v=v_u+2.0*(v_c-v_u)
        z=restore(z+(float(ts[i+1])-tc)*v,x0,prompt); xp=x_c
    return nrm(z-x0)/triv

r=np.array([once(s) for s in range(a.reps)])
print(f"ckpt {ta['model']} @ {ta['steps']} steps, weights={a.weights}, {a.reps} repeats")
print(f"  z_err_ratio  mean {r.mean():.4f}  std {r.std():.4f}  min {r.min():.4f}  max {r.max():.4f}")
print(f"  spread (max-min)        {r.max()-r.min():.4f}")
print(f"  1 sigma                 {r.std():.4f}")
print()
obs=[1.116,1.104,1.087,1.074,1.079]
print("observed eval points:", obs)
print(f"  largest step-to-step DROP  {max(obs[i]-obs[i+1] for i in range(4)):+.4f}")
print(f"  the rise at 25k            {obs[4]-obs[3]:+.4f}")
print(f"\n  -> the rise is {'WITHIN' if obs[4]-obs[3] < 2*r.std() else 'OUTSIDE'} 2 sigma, "
      f"so it is {'noise' if obs[4]-obs[3] < 2*r.std() else 'a real stall'}")
json.dump({"ratios":r.tolist(),"mean":float(r.mean()),"std":float(r.std())},
          open(f"{HERE}/out/eval_noise.json","w"),indent=2)
