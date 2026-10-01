#!/usr/bin/env python
"""ELF-TTS on LJSpeech — training, sampling and the path-A/path-B comparison.

Faithful to ELF's train_step: three forwards per iteration (two no_grad for the
self-conditioning guided target, one with gradient), a per-example Bernoulli draw
routing rows to the CE or L2 branch, and a single loss denominator.

TTS substitutions:
  x0            WavTokenizer pre-quantization latent, /latent_std
  text          character ids average-upsampled to the frame grid, entering as a
                third channel block alongside [z, x_pred]
  cond region   the audio prompt (frames [0,P)), marked by prompt_mask, which is
                what ELF's add_noise / restore_cond / loss_mask already protect
  CE target     the single-codebook id in [0, 4096)
  CFG drop      zeroes the text block (stock ELF zeroes condition positions)
"""
import argparse, json, math, os, pickle, sys, time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from elf_tts.model import ELF_models  # noqa: E402

T_EPS = 5e-2


def log(*a):
    print(*a, flush=True)


# ============================================================ model wrapper
class ELFTTS(nn.Module):
    """ELF plus the character embedding that feeds its third channel block."""

    def __init__(self, name, char_vocab, audio_vocab, dim=512, max_length=400,
                 cb_skip=False, **kw):
        super().__init__()
        self.dim = dim
        self.cb_skip = cb_skip
        self.char_embed = nn.Embedding(char_vocab, dim, padding_idx=0)
        nn.init.normal_(self.char_embed.weight, std=0.02)
        with torch.no_grad():
            self.char_embed.weight[0].zero_()
        self.elf = ELF_models[name](
            text_encoder_dim=dim, max_length=max_length, vocab_size=audio_vocab,
            n_input_blocks=3, bottleneck_dim=128, num_time_tokens=4,
            num_self_cond_cfg_tokens=4, num_model_mode_tokens=4, **kw)
        # Frozen nearest-codebook logits, as buffers. argmin_k||z-c_k||^2 ==
        # argmax_k(c_k.z - ||c_k||^2/2), so this layer reproduces path A exactly.
        self.register_buffer("cb_W", torch.zeros(dim, audio_vocab), persistent=False)
        self.register_buffer("cb_b", torch.zeros(audio_vocab), persistent=False)

    def set_codebook(self, codebook, latent_std, init_unembed=False):
        """codebook: (K, D) raw codec codebook. Scaled into /latent_std units."""
        cb = codebook.to(self.cb_W.device).float() / latent_std
        self.cb_W.copy_(cb.t())
        self.cb_b.copy_(-0.5 * cb.pow(2).sum(1))
        if init_unembed:
            with torch.no_grad():
                self.elf.unembed_kernel.copy_(self.cb_W)
                self.elf.unembed_bias.copy_(self.cb_b)
        if self.cb_skip:
            # Start the learned head as an exact zero residual, so the model
            # begins at path A and can only improve on it.
            with torch.no_grad():
                self.elf.unembed_kernel.zero_()
                self.elf.unembed_bias.zero_()

    def forward(self, z, sc, text_ids, t, *, text_emb=None, attention_mask=None,
                self_cond_cfg_scale=None, decoder_step_active=None, drop_text=None):
        """z, sc: (B,T,D). text_ids: (B,T) long. drop_text: (B,) 1 = unconditional."""
        if text_emb is None:
            text_emb = self.char_embed(text_ids)
        if drop_text is not None:
            text_emb = text_emb * (1.0 - drop_text.to(text_emb.dtype).view(-1, 1, 1))
        x = torch.cat([z, sc, text_emb], dim=-1)
        net_out, logits = self.elf(
            x, t, attention_mask=attention_mask, deterministic=not self.training,
            self_cond_cfg_scale=self_cond_cfg_scale,
            decoder_step_active=decoder_step_active)
        if self.cb_skip and logits is not None:
            # Learned head becomes a residual correction on nearest-codebook.
            logits = logits + (z.float() @ self.cb_W + self.cb_b)
        return net_out, logits


# ============================================================ data
class LJCache(Dataset):
    def __init__(self, cache_dir, split, val_n=500, test_n=500, seed=0):
        with open(os.path.join(cache_dir, "index.pt"), "rb") as f:
            self.meta = pickle.load(f)
        self.recs = self.meta["records"]
        self.total = self.meta["total_frames"]
        self.lat = np.memmap(os.path.join(cache_dir, "latents.f16"),
                             dtype=np.float16, mode="r", shape=(self.total, 512))
        idx = np.arange(len(self.recs))
        np.random.default_rng(seed).shuffle(idx)
        # StyleTTS2-style division: held-out test, small val, rest train
        self.ids = {"test": idx[:test_n], "val": idx[test_n:test_n + val_n],
                    "train": idx[test_n + val_n:]}[split]
        self.split = split

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        r = self.recs[self.ids[i]]
        o, T = r["offset"], r["T"]
        return {
            "latent": np.asarray(self.lat[o:o + T], dtype=np.float32),
            "code": r["code"].astype(np.int64),
            "chars": r["chars"].astype(np.int64),
            "id": r["id"],
        }


def make_collate(max_length, prompt_frac, rng_seed=0, fixed_prompt=None):
    """Average upsampling + prompt masking.

    text_ids[j] = chars[floor(j * N / T)] over the clip's real length T, so the
    prompt's own transcript lands on [0,P) and the target text on [P,T).
    """
    rng = np.random.default_rng(rng_seed)

    def collate(batch):
        B = len(batch)
        L = max_length
        lat = np.zeros((B, L, 512), dtype=np.float32)
        code = np.zeros((B, L), dtype=np.int64)
        text = np.zeros((B, L), dtype=np.int64)
        valid = np.zeros((B, L), dtype=np.float32)
        prompt = np.zeros((B, L), dtype=np.float32)
        ids, Ts, Ps = [], [], []
        for b, it in enumerate(batch):
            T = min(it["latent"].shape[0], L)
            N = max(len(it["chars"]), 1)
            lat[b, :T] = it["latent"][:T]
            code[b, :T] = it["code"][:T]
            # average upsampling over the clip's own length
            j = np.arange(T)
            text[b, :T] = it["chars"][np.minimum((j * N) // T, N - 1)]
            valid[b, :T] = 1.0
            pf = fixed_prompt if fixed_prompt is not None else rng.uniform(*prompt_frac)
            P = int(round(T * pf))
            prompt[b, :P] = 1.0
            ids.append(it["id"]); Ts.append(T); Ps.append(P)
        return {
            "latent": torch.from_numpy(lat), "code": torch.from_numpy(code),
            "text": torch.from_numpy(text), "valid": torch.from_numpy(valid),
            "prompt": torch.from_numpy(prompt), "id": ids,
            "T": torch.tensor(Ts), "P": torch.tensor(Ps),
        }
    return collate


# ============================================================ ELF math
def sample_timesteps(n, p_mean, p_std, device, dtype=torch.float32):
    return torch.sigmoid(torch.randn((n,), device=device, dtype=dtype) * p_std + p_mean)


def sample_cfg_scale(n, lo, hi, device, dtype=torch.float32):
    u = torch.rand((n,), device=device, dtype=dtype)
    a, b = 1.0 + lo, 1.0 + hi
    return a * torch.exp(u * math.log(b / a)) - 1.0


def get_sampling_steps(n_steps, schedule, p_mean, p_std, device):
    if schedule == "uniform":
        return torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    s = torch.sort(sample_timesteps(n_steps - 1, p_mean, p_std, device)).values
    return torch.cat([torch.zeros(1, device=device), s, torch.ones(1, device=device)])


def restore(z, clean, mask):
    """Put the clean prompt latents back at prompt positions."""
    return torch.where(mask.unsqueeze(-1) > 0, clean, z)


def net_to_v(net_out, z, t):
    x = net_out[0] if isinstance(net_out, tuple) else net_out
    return (x - z) / torch.clamp(1.0 - t.view(-1, 1, 1), min=T_EPS), x


# ============================================================ train step
def train_step(model, ema, batch, cfg, opt, device, scaler=None):
    x0 = (batch["latent"].to(device, non_blocking=True) / cfg.latent_std)
    code = batch["code"].to(device, non_blocking=True)
    text = batch["text"].to(device, non_blocking=True)
    valid = batch["valid"].to(device, non_blocking=True)
    prompt = batch["prompt"].to(device, non_blocking=True)
    B, L, D = x0.shape

    text_emb = model.char_embed(text)
    drop_text = (torch.rand(B, device=device) < cfg.label_drop_prob).float()
    text_emb = text_emb * (1.0 - drop_text.view(-1, 1, 1))

    t = sample_timesteps(B, cfg.denoiser_p_mean, cfg.denoiser_p_std, device)
    noise = torch.randn_like(x0)
    z = t.view(-1, 1, 1) * x0 + (1 - t.view(-1, 1, 1)) * noise * cfg.denoiser_noise_scale
    z = restore(z, x0, prompt)                               # prompt stays clean

    loss_mask = valid * (1.0 - prompt)                       # neither loss sees the prompt

    # per-example branch draw
    d = torch.bernoulli(torch.full((B,), cfg.decoder_prob, device=device))
    lam = torch.sigmoid(torch.randn((B, L, 1), device=device) * cfg.decoder_p_std + cfg.decoder_p_mean)
    dec_noise = torch.randn_like(x0) * cfg.decoder_noise_scale
    decoder_z = lam * x0 + (1 - lam) * dec_noise
    z_mixed = d.view(-1, 1, 1) * decoder_z + (1 - d.view(-1, 1, 1)) * z
    t_mixed = d * torch.ones_like(t) + (1 - d) * t

    v_target = (x0 - z) / torch.clamp(1 - t.view(-1, 1, 1), min=T_EPS)

    sc_cfg = sample_cfg_scale(B, cfg.self_cond_cfg_min, cfg.self_cond_cfg_max, device)
    use_sc = (torch.rand(B, device=device) < cfg.self_cond_prob).float().view(-1, 1, 1)

    # ---- two no_grad forwards build the guided target
    with torch.no_grad():
        zeros_sc = restore(torch.zeros_like(z), x0, prompt)
        out_u = model(z, zeros_sc, text, t, text_emb=text_emb,
                      self_cond_cfg_scale=sc_cfg)
        v_u, x_u = net_to_v(out_u, z, t)
        x_u = restore(x_u, x0, prompt)
        out_c = model(z, x_u, text, t, text_emb=text_emb, self_cond_cfg_scale=sc_cfg)
        v_c, _ = net_to_v(out_c, z, t)
        guidance = (1 - 1 / sc_cfg.view(-1, 1, 1)) * (v_c - v_u)
        v_final = (v_target + torch.where(use_sc.bool(), guidance, torch.zeros_like(guidance))).detach()
        sc_half = restore(x_u * use_sc, x0, prompt) * (1 - d.view(-1, 1, 1))

    # ---- the gradient forward
    net_out, logits = model(z_mixed, sc_half, text, t_mixed, text_emb=text_emb,
                            self_cond_cfg_scale=sc_cfg, decoder_step_active=d)

    v_pred, _ = net_to_v(net_out, z, t)
    l2 = ((v_pred - v_final) ** 2).mean(-1)
    ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                         code.reshape(-1), reduction="none").view(B, L)

    m_ce = loss_mask * d.view(-1, 1)
    m_l2 = loss_mask * (1 - d).view(-1, 1)
    denom = loss_mask.sum().clamp(min=1.0)
    loss = ((ce * m_ce).sum() + (l2 * m_l2).sum()) / denom

    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    with torch.no_grad():
        for pe, p in zip(ema.parameters(), model.parameters()):
            pe.mul_(cfg.ema_decay).add_(p.detach(), alpha=1 - cfg.ema_decay)

    with torch.no_grad():
        acc = ((logits.argmax(-1) == code).float() * m_ce).sum() / m_ce.sum().clamp(min=1)
    return {
        "loss": loss.item(),
        "l2": ((l2 * m_l2).sum() / m_l2.sum().clamp(min=1)).item(),
        "ce": ((ce * m_ce).sum() / m_ce.sum().clamp(min=1)).item(),
        "ce_acc": acc.item(), "gnorm": gn.item(),
    }


# ============================================================ eval
def load_codebook(ckpt_path, device):
    """Pull just the 4096x512 codebook out of the codec checkpoint — no codec needed."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    k = "feature_extractor.encodec.quantizer.vq.layers.0._codebook.embed"
    return sd[k].to(device).float()


@torch.no_grad()
def path_a_curve(codebook, loader, cfg, device, sigmas, max_batches=4):
    """Path A: nearest-codebook survival, on the same frames/batches as path B.

    Latents are in /latent_std units here, so the codebook is scaled to match.
    """
    cb = codebook / cfg.latent_std
    cb_sq = cb.pow(2).sum(1)
    keep = np.zeros(len(sigmas)); tot = 0.0
    for bi, b in enumerate(loader):
        if bi >= max_batches:
            break
        x0 = b["latent"].to(device) / cfg.latent_std
        m = (b["valid"].to(device) * (1 - b["prompt"].to(device)))
        code = b["code"].to(device)
        flat = x0.reshape(-1, x0.shape[-1]); mf = m.reshape(-1); cf = code.reshape(-1)
        for si, s in enumerate(sigmas):
            xx = flat + torch.randn_like(flat) * s
            d = xx.pow(2).sum(1, keepdim=True) - 2 * xx @ cb.t() + cb_sq[None, :]
            keep[si] += ((d.argmin(1) == cf).float() * mf).sum().item()
        tot += mf.sum().item()
    return (keep / max(tot, 1)).tolist()


@torch.no_grad()
def path_b_curve(model, loader, cfg, device, sigmas, max_batches=4):
    """Path B: feed x0 + sigma*noise to the CE head at t=1 and measure code survival.

    Directly comparable to Test 0's path A (nearest-codebook), which used the same
    sigma grid in units of latent_std.
    """
    model.eval()
    keep = np.zeros(len(sigmas)); tot = 0.0
    for bi, b in enumerate(loader):
        if bi >= max_batches:
            break
        x0 = b["latent"].to(device) / cfg.latent_std
        code = b["code"].to(device); text = b["text"].to(device)
        m = (b["valid"].to(device) * (1 - b["prompt"].to(device)))
        B = x0.shape[0]
        t1 = torch.ones(B, device=device)
        sc = torch.ones(B, device=device)
        for si, s in enumerate(sigmas):
            # sigma is in units of latent_std; x0 here is already /latent_std
            zz = x0 + torch.randn_like(x0) * s
            _, logits = model(zz, torch.zeros_like(zz), text, t1,
                              self_cond_cfg_scale=sc, decoder_step_active=True)
            keep[si] += ((logits.argmax(-1) == code).float() * m).sum().item()
        tot += m.sum().item()
    return (keep / max(tot, 1)).tolist()


@torch.no_grad()
def convergence_metrics(model, loader, cfg, device, n_steps=32, cfg_scale=2.0,
                        schedule="logit_normal", max_batches=2):
    """The measurements that tell slow convergence apart from a structural failure.

    conditional_sharpness : ||x0_hat|| / ||x0|| at low t. The model's x0 prediction at
        t=0.05 was measured at 0.55 of the true norm -- a blurred conditional mean,
        the signature of a model that cannot yet resolve WHICH utterance the text
        specifies. If this ratio climbs toward 1.0 over training, it is converging.
        If it plateaus, the model is stuck at the conditional mean.

    exposure_gap : ||x0_hat - x0|| on the rollout's own z, divided by the same error on
        a TRUE z_t at the same t. 1.0 means the rollout stays on-manifold; it measured
        11.6x at t=0.7. Closing over training = convergence.

    rollout_z_err : ||z_final - x0||. Must fall below ~10 for the CE head to work.

    text_sensitivity : ||x0_hat(text) - x0_hat(no text)|| / ||x0_hat|| at low t.
        Is the model learning to use the conditioning at all?
    """
    model.eval()
    out = {}
    tf_err, ro_err = {}, {}
    sharp, tsens = {}, {}
    probe_ts = (0.05, 0.3, 0.7)

    # --- teacher-forced, on true z_t
    for tv in probe_ts:
        e = s_ = ts_ = 0.0; nb = 0
        for bi, b in enumerate(loader):
            if bi >= max_batches:
                break
            x0 = b["latent"].to(device) / cfg.latent_std
            text = b["text"].to(device); prompt = b["prompt"].to(device)
            m = b["valid"].to(device) * (1 - prompt)
            B = x0.shape[0]
            t = torch.full((B,), tv, device=device)
            z = restore(t.view(-1, 1, 1) * x0 +
                        (1 - t.view(-1, 1, 1)) * torch.randn_like(x0) * cfg.denoiser_noise_scale,
                        x0, prompt)
            sc = torch.ones(B, device=device)
            zsc = restore(torch.zeros_like(z), x0, prompt)
            xh = model(z, zsc, text, t, self_cond_cfg_scale=sc,
                       drop_text=torch.zeros(B, device=device))[0]
            xh0 = model(z, zsc, text, t, self_cond_cfg_scale=sc,
                        drop_text=torch.ones(B, device=device))[0]
            nrm = lambda a: ((a.norm(dim=-1) * m).sum() / m.sum().clamp(min=1)).item()
            e += nrm(xh - x0); s_ += nrm(xh) / max(nrm(x0), 1e-9); ts_ += nrm(xh - xh0) / max(nrm(xh), 1e-9)
            nb += 1
        tf_err[tv] = e / nb; sharp[tv] = s_ / nb; tsens[tv] = ts_ / nb

    # --- rollout, recording x0_hat error at the same probe points
    b = next(iter(loader))
    x0 = b["latent"].to(device) / cfg.latent_std
    text = b["text"].to(device); prompt = b["prompt"].to(device)
    m = b["valid"].to(device) * (1 - prompt)
    B = x0.shape[0]
    nrm = lambda a: ((a.norm(dim=-1) * m).sum() / m.sum().clamp(min=1)).item()
    z = restore(torch.randn_like(x0) * cfg.denoiser_noise_scale, x0, prompt)
    x_pred = restore(torch.zeros_like(z), x0, prompt)
    ts = get_sampling_steps(n_steps, schedule, cfg.denoiser_p_mean, cfg.denoiser_p_std, device)
    scb = torch.ones(B, device=device)
    zd, od = torch.zeros(B, device=device), torch.ones(B, device=device)
    for i in range(len(ts) - 1):
        t = ts[i].expand(B)
        out_c = model(z, x_pred, text, t, self_cond_cfg_scale=scb, drop_text=zd)
        v_c, x_c = net_to_v(out_c, z, t)
        v_c = restore(v_c, torch.zeros_like(z), prompt); x_c = restore(x_c, x0, prompt)
        if cfg_scale != 1.0:
            v_u, _ = net_to_v(model(z, x_pred, text, t, self_cond_cfg_scale=scb, drop_text=od), z, t)
            v_u = restore(v_u, torch.zeros_like(z), prompt)
            v = v_u + cfg_scale * (v_c - v_u)
        else:
            v = v_c
        z = restore(z + (ts[i + 1] - ts[i]) * v, x0, prompt)
        x_pred = x_c
        tcur = float(ts[i])
        for tv in probe_ts:
            if tv not in ro_err or abs(tcur - tv) < abs(ro_err[tv][0] - tv):
                ro_err[tv] = (tcur, nrm(x_c - x0))

    out["rollout/z_err"] = nrm(z - x0)
    out["rollout/x0hat_err"] = nrm(x_pred - x0)
    out["ref/x0_norm"] = nrm(x0)
    for tv in probe_ts:
        out[f"tf/x0hat_err_t{tv}"] = tf_err[tv]
        out[f"sharpness/ratio_t{tv}"] = sharp[tv]
        out[f"text_sens/rel_t{tv}"] = tsens[tv]
        if tv in ro_err:
            out[f"rollout/x0hat_err_t{tv}"] = ro_err[tv][1]
            out[f"exposure_gap/t{tv}"] = ro_err[tv][1] / max(tf_err[tv], 1e-9)
    return out


@torch.no_grad()
def sample(model, batch, cfg, device, n_steps=32, cfg_scale=2.0, sc_cfg=1.0,
           schedule="logit_normal"):
    """ODE rollout then one decode forward. Returns predicted code ids."""
    model.eval()
    x0 = batch["latent"].to(device) / cfg.latent_std
    text = batch["text"].to(device)
    prompt = batch["prompt"].to(device)
    B, L, D = x0.shape

    z = torch.randn_like(x0) * cfg.denoiser_noise_scale
    z = restore(z, x0, prompt)
    x_pred = restore(torch.zeros_like(z), x0, prompt)
    ts = get_sampling_steps(n_steps, schedule, cfg.denoiser_p_mean, cfg.denoiser_p_std, device)
    scb = torch.full((B,), float(sc_cfg), device=device)
    zero_drop = torch.zeros(B, device=device)
    one_drop = torch.ones(B, device=device)

    nfe = 0
    for i in range(len(ts) - 1):
        t = ts[i].expand(B)
        out_c = model(z, x_pred, text, t, self_cond_cfg_scale=scb, drop_text=zero_drop)
        v_c, x_c = net_to_v(out_c, z, t); nfe += 1
        v_c, x_c = restore(v_c, torch.zeros_like(z), prompt), restore(x_c, x0, prompt)
        if cfg_scale != 1.0:
            out_u = model(z, x_pred, text, t, self_cond_cfg_scale=scb, drop_text=one_drop)
            v_u, _ = net_to_v(out_u, z, t); nfe += 1
            v_u = restore(v_u, torch.zeros_like(z), prompt)
            v = v_u + cfg_scale * (v_c - v_u)
        else:
            v = v_c
        z = z + (ts[i + 1] - ts[i]) * v
        z = restore(z, x0, prompt)
        x_pred = x_c

    t1 = torch.ones(B, device=device)
    _, logits = model(z, torch.zeros_like(z), text, t1,
                      self_cond_cfg_scale=scb, decoder_step_active=True)
    nfe += 1
    return logits.argmax(-1), z, nfe


@torch.no_grad()
def eval_sampled(model, loader, cfg, device, max_batches=2, **kw):
    """End-to-end: sample, then compare predicted codes with ground truth."""
    hit = tot = 0.0
    nfe = 0
    for bi, b in enumerate(loader):
        if bi >= max_batches:
            break
        pred, _, nfe = sample(model, b, cfg, device, **kw)
        code = b["code"].to(device)
        m = (b["valid"].to(device) * (1 - b["prompt"].to(device)))
        hit += ((pred == code).float() * m).sum().item(); tot += m.sum().item()
    return hit / max(tot, 1), nfe


# ============================================================ main
@dataclass
class Cfg:
    latent_std: float = 0.26409
    denoiser_p_mean: float = -1.5
    denoiser_p_std: float = 0.8
    denoiser_noise_scale: float = 2.0
    decoder_prob: float = 0.2
    decoder_p_mean: float = 0.8
    decoder_p_std: float = 0.8
    decoder_noise_scale: float = 1.0
    self_cond_prob: float = 0.5
    self_cond_cfg_min: float = 0.5
    self_cond_cfg_max: float = 5.0
    label_drop_prob: float = 0.20
    ema_decay: float = 0.999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(HERE, "cache"))
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "test1"))
    ap.add_argument("--model", default="ELF-T")
    ap.add_argument("--max_length", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--overfit", type=int, default=0, help="N clips to overfit (Test 1)")
    ap.add_argument("--no_text", action="store_true", help="Test 1: zero the text block")
    ap.add_argument("--prompt_lo", type=float, default=0.0)
    ap.add_argument("--prompt_hi", type=float, default=0.3)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ema_decay", type=float, default=0.999,
                    help="0.999 leaves 13% of init in the EMA at 2k steps — raise for long runs")
    ap.add_argument("--codec_ckpt", default=os.path.join(
        HERE, "ckpt", "wavtokenizer_large_unify_600_24k.ckpt"))
    ap.add_argument("--init_unembed", action="store_true",
                    help="initialise unembed_kernel/bias from the codec codebook")
    ap.add_argument("--decoder_prob", type=float, default=0.2,
                    help="CE-branch fraction. ELF default 0.5; de-en config uses 0.2")
    ap.add_argument("--sample_schedule", default="logit_normal",
                    choices=["logit_normal", "uniform"])
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="voiceflow")
    ap.add_argument("--wandb_run", default=None)
    ap.add_argument("--cb_skip", action="store_true",
                    help="logits = frozen nearest-codebook logits + learned residual")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Cfg()
    cfg.ema_decay = args.ema_decay
    cfg.decoder_prob = args.decoder_prob
    if args.no_text:
        cfg.label_drop_prob = 1.0            # text block always zero
    log(f"device {device} | model {args.model} | steps {args.steps} | batch {args.batch}")
    log(f"latent_std {cfg.latent_std} | label_drop {cfg.label_drop_prob} | "
        f"decoder_prob {cfg.decoder_prob} | sample_schedule {args.sample_schedule}")

    wb = None
    if args.wandb:
        import wandb as _wb
        wb = _wb
        wb.init(project=args.wandb_project,
                name=args.wandb_run or os.path.basename(args.out),
                config={**vars(args), **cfg.__dict__})
        log(f"wandb: {wb.run.url}")

    train_ds = LJCache(args.cache, "train", seed=args.seed)
    val_ds = LJCache(args.cache, "val", seed=args.seed)
    test_ds = LJCache(args.cache, "test", seed=args.seed)
    meta = train_ds.meta
    log(f"splits: train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)}")
    log(f"char vocab {meta['char_vocab_size']} | audio vocab {meta['audio_vocab_size']}")
    log(f"frames/char {meta['frames_per_char']:.2f}")

    if args.overfit:
        train_ds.ids = train_ds.ids[:args.overfit]
        val_ds = train_ds                      # memorisation check
        log(f"OVERFIT mode: {len(train_ds)} clips, val == train")

    coll = make_collate(args.max_length, (args.prompt_lo, args.prompt_hi), args.seed)
    coll_eval = make_collate(args.max_length, (0.3, 0.3), args.seed, fixed_prompt=0.3)
    dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True,
                    num_workers=args.workers, collate_fn=coll, pin_memory=True,
                    persistent_workers=args.workers > 0)
    dl_val = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=0, collate_fn=coll_eval)
    # prompt = 0: no ground-truth audio at all. This is PURE text-to-speech, and it is
    # in-distribution because training draws prompt_frac ~ U(0.0, 0.3).
    coll_p0 = make_collate(args.max_length, (0.0, 0.0), args.seed, fixed_prompt=0.0)
    dl_val_p0 = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                           num_workers=0, collate_fn=coll_p0)

    codebook = load_codebook(args.codec_ckpt, device)
    model = ELFTTS(args.model, meta["char_vocab_size"], meta["audio_vocab_size"],
                   dim=512, max_length=args.max_length, cb_skip=args.cb_skip).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    n_elf = sum(p.numel() for p in model.elf.parameters())
    log(f"params total {n_par/1e6:.2f}M  (elf {n_elf/1e6:.2f}M + char_embed "
        f"{model.char_embed.weight.numel()/1e6:.3f}M)")

    ema = ELFTTS(args.model, meta["char_vocab_size"], meta["audio_vocab_size"],
                 dim=512, max_length=args.max_length, cb_skip=args.cb_skip).to(device)
    model.set_codebook(codebook, cfg.latent_std, init_unembed=args.init_unembed)
    ema.load_state_dict(model.state_dict())
    ema.set_codebook(codebook, cfg.latent_std)
    log(f"codebook wired | cb_skip={args.cb_skip} | init_unembed={args.init_unembed}")
    for p in ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(args.warmup, 1)))

    sigmas = [0.0, 0.01, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00]
    sig_half = sigmas.index(0.50)
    pa = path_a_curve(codebook, dl_val, cfg, device, sigmas)
    log("path A (nearest-codebook) on the eval batches: " +
        " ".join(f"{s:.2f}:{v*100:.1f}%" for s, v in zip(sigmas, pa)))
    hist = []
    it = iter(dl)
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl); b = next(it)
        m = train_step(model, ema, b, cfg, opt, device)
        sched.step()
        if step % args.log_every == 0 or step == 1:
            el = time.time() - t0
            log(f"step {step:>6} | loss {m['loss']:.4f} | l2 {m['l2']:.4f} | "
                f"ce {m['ce']:.4f} | ce_acc {m['ce_acc']*100:5.2f}% | "
                f"gn {m['gnorm']:.2f} | {step/el:.2f} it/s")
            hist.append({"step": step, **m, "sec": el})
            if wb:
                wb.log({f"train/{k}": v for k, v in m.items()} |
                       {"train/lr": opt.param_groups[0]["lr"], "it_per_s": step / el},
                       step=step)
        if step % args.eval_every == 0 or step == args.steps:
            rec = {"step": step, "path_sigmas": sigmas, "path_a_survival": pa}
            for tag, net in (("raw", model), ("ema", ema)):
                pb = path_b_curve(net, dl_val, cfg, device, sigmas)
                rec[f"path_b_survival_{tag}"] = pb
                ret_a = pa[sig_half] / max(pa[0], 1e-9)
                ret_b = pb[sig_half] / max(pb[0], 1e-9)
                line = (f"  [eval @ {step} | {tag}] pathB sigma0 {pb[0]*100:5.2f}%  "
                        f"sigma0.5 {pb[sig_half]*100:5.2f}%  retention {ret_b*100:5.1f}%"
                        f"   (pathA {pa[0]*100:5.2f}% / {pa[sig_half]*100:5.2f}% "
                        f"retention {ret_a*100:5.1f}%)")
                if not args.no_text:
                    acc, nfe = eval_sampled(net, dl_val, cfg, device, n_steps=32,
                                            cfg_scale=2.0, schedule=args.sample_schedule)
                    rec[f"sampled_acc_{tag}"] = acc; rec["nfe"] = nfe
                    line += f"  | sampled code acc {acc*100:.2f}% (NFE {nfe})"
                log(line)
            cm = {}
            for tag, net in (("raw", model), ("ema", ema)):
                for reg, loader in (("p30", dl_val), ("p0", dl_val_p0)):
                    c = convergence_metrics(net, loader, cfg, device,
                                            schedule=args.sample_schedule)
                    rec[f"conv_{tag}_{reg}"] = c
                    cm.update({f"{k}/{reg}/{tag}": v for k, v in c.items()})
                    a, _ = eval_sampled(net, loader, cfg, device, n_steps=32,
                                        cfg_scale=2.0, schedule=args.sample_schedule)
                    rec[f"sampled_acc_{tag}_{reg}"] = a
                    cm[f"sampled/code_acc/{reg}/{tag}"] = a
                    log(f"  [conv @ {step} | {tag} | {reg}] "
                        f"code acc {a*100:5.2f}% | rollout ||z-x0|| {c['rollout/z_err']:6.2f} "
                        f"| sharpness t0.05 {c['sharpness/ratio_t0.05']:.3f} "
                        f"| exposure gap t0.7 {c.get('exposure_gap/t0.7', float('nan')):5.1f}x "
                        f"| text sens t0.05 {c['text_sens/rel_t0.05']*100:4.1f}%")
            if wb:
                wl = dict(cm)
                for tag in ("raw", "ema"):
                    if f"path_b_survival_{tag}" in rec:
                        pb = rec[f"path_b_survival_{tag}"]
                        wl[f"pathB/sigma0/{tag}"] = pb[0]
                        wl[f"pathB/sigma0.5/{tag}"] = pb[sig_half]
                    if f"sampled_acc_{tag}" in rec:
                        wl[f"sampled/code_acc/{tag}"] = rec[f"sampled_acc_{tag}"]
                wl["pathA/sigma0"] = pa[0]; wl["pathA/sigma0.5"] = pa[sig_half]
                wb.log(wl, step=step)
            hist.append(rec)
            model.train()

    torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                "args": vars(args), "cfg": cfg.__dict__, "meta_keys": list(meta.keys())},
               os.path.join(args.out, "ckpt.pt"))
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump({"history": hist, "args": vars(args), "cfg": cfg.__dict__,
                   "params_M": n_par / 1e6}, f, indent=2)
    log(f"saved {args.out}/ckpt.pt and history.json")
    log(f"total {time.time()-t0:.0f}s for {args.steps} steps")
    if wb:
        wb.finish()


if __name__ == "__main__":
    main()
