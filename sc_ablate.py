#!/usr/bin/env python
"""Does self-conditioning actually do anything in our sampler?

Structurally x_pred IS fed and carried forward. This measures whether it HELPS.
Grid: SC on/off x sc_cfg x cfg. If 'SC off' matches or beats 'SC on', the
self-conditioning input is not being used the way training set it up.
"""
import os, sys, json, torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from train import (ELFTTS, LJCache, make_collate, Cfg, load_codebook,
                   get_sampling_steps, restore, net_to_v)
from torch.utils.data import DataLoader

dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck=torch.load(f"{HERE}/runs/test2/ckpt.pt",map_location="cpu",weights_only=False)
ta=ck["args"]; cfg=Cfg(**{k:v for k,v in ck["cfg"].items() if k in Cfg.__dataclass_fields__})
val=LJCache(f"{HERE}/cache","val",seed=ta["seed"]); meta=val.meta
m_=ELFTTS(ta["model"],meta["char_vocab_size"],meta["audio_vocab_size"],dim=512,
          max_length=ta["max_length"]).to(dev); m_.load_state_dict(ck["ema"]); m_.eval()
cb=load_codebook(f"{HERE}/ckpt/wavtokenizer_large_unify_600_24k.ckpt",dev)/cfg.latent_std
cbsq=cb.pow(2).sum(1)

def nn_codes(z):
    f=z.reshape(-1,z.shape[-1])
    return (f.pow(2).sum(1,keepdim=True)-2*f@cb.t()+cbsq[None,:]).argmin(1).view(z.shape[:-1])

@torch.no_grad()
def run(b, use_sc, sc_cfg, cfg_scale, n_steps=32, seed=0):
    torch.manual_seed(seed)
    x0=b["latent"].to(dev)/cfg.latent_std; code=b["code"].to(dev)
    text=b["text"].to(dev); prompt=b["prompt"].to(dev)
    msk=b["valid"].to(dev)*(1-prompt); B=x0.shape[0]
    nrm=lambda a:((a.norm(dim=-1)*msk).sum()/msk.sum().clamp(min=1)).item()
    z=restore(torch.randn_like(x0)*cfg.denoiser_noise_scale,x0,prompt)
    zero_sc=restore(torch.zeros_like(z),x0,prompt)
    x_pred=zero_sc
    ts=get_sampling_steps(n_steps,"logit_normal",cfg.denoiser_p_mean,cfg.denoiser_p_std,dev)
    scb=torch.full((B,),float(sc_cfg),device=dev)
    zd,od=torch.zeros(B,device=dev),torch.ones(B,device=dev)
    for i in range(len(ts)-1):
        t=ts[i].expand(B)
        sc_in = x_pred if use_sc else zero_sc
        v_c,x_c=net_to_v(m_(z,sc_in,text,t,self_cond_cfg_scale=scb,drop_text=zd),z,t)
        v_c=restore(v_c,torch.zeros_like(z),prompt); x_c=restore(x_c,x0,prompt)
        if cfg_scale!=1.0:
            v_u,_=net_to_v(m_(z,sc_in,text,t,self_cond_cfg_scale=scb,drop_text=od),z,t)
            v_u=restore(v_u,torch.zeros_like(z),prompt)
            v=v_u+cfg_scale*(v_c-v_u)
        else: v=v_c
        z=restore(z+(ts[i+1]-ts[i])*v,x0,prompt); x_pred=x_c
    t1=torch.ones(B,device=dev)
    _,lg=m_(z,torch.zeros_like(z),text,t1,self_cond_cfg_scale=scb,decoder_step_active=True)
    acc=lambda p:(((p==code).float()*msk).sum()/msk.sum().clamp(min=1)).item()
    return nrm(z-x0), acc(lg.argmax(-1)), acc(nn_codes(z))

coll=make_collate(ta["max_length"],(0.0,0.0),ta["seed"],fixed_prompt=0.0)
b=next(iter(DataLoader(val,batch_size=16,shuffle=False,num_workers=0,collate_fn=coll)))
print(f"ckpt {ta['model']} @ {ta['steps']} steps, ema | pure TTS (prompt=0), 32 steps, logit_normal")
print(f"{'SC':>5} {'sc_cfg':>7} {'cfg':>5} | {'||z-x0||':>9} {'CE acc':>8} {'NN acc':>8}")
print("-"*58)
out=[]
for use_sc in (True,False):
    for sc_cfg in (1.0,2.0,3.0):
        for cg in (1.0,2.0):
            e,a,an=run(b,use_sc,sc_cfg,cg)
            print(f"{'on' if use_sc else 'OFF':>5} {sc_cfg:>7.1f} {cg:>5.1f} | {e:>9.3f} {a*100:>7.2f}% {an*100:>7.2f}%")
            out.append(dict(use_sc=use_sc,sc_cfg=sc_cfg,cfg=cg,z_err=e,ce_acc=a,nn_acc=an))
json.dump(out,open(f"{HERE}/out/sc_ablate.json","w"),indent=2)
print("\nIf 'OFF' rows match or beat 'on' rows, the self-conditioning input is inert.")
