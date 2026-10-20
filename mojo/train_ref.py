"""Autograd reference for Timestral-209M fine-tuning (univariate, context CTX log returns -> 32-step x 21-quantile head).

Params are fp32 master leaves; compute dtype is fp32 (oracle) or bf16 (mirrors the Mojo path: bf16 GEMM operands/outputs,
fp32 residual stream, norms and softmax). Loss: pinball in denormalised log-return space, step 0 weight 1, steps 1..31 weight aux/31.
"""
import math, numpy as np, torch
import torch.nn.functional as F

H, D, NH, HD, FF, PS = 768, 768, 12, 64, 2048, 32
SCHED = ["time", "time", "time", "time_variate"] * 6
LEVELS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
LOSS_SCALE = 0.02


def build_params(model, device="cuda"):
    sd = {k: v.detach().to(device).float() for k, v in model.state_dict().items()}
    def reorder_qkv(w): return w.reshape(3, HD, NH, H).permute(0, 2, 1, 3).reshape(3 * D, H)
    P = {
        "fq": sd["representation.forecast_query"],
        "e1": sd["representation.patch_proj.hidden_layer.weight"][:, :PS], "e1b": sd["representation.patch_proj.hidden_layer.bias"],
        "e2": sd["representation.patch_proj.output_layer.weight"], "e2b": sd["representation.patch_proj.output_layer.bias"],
        "er": sd["representation.patch_proj.residual_layer.weight"][:, :PS], "erb": sd["representation.patch_proj.residual_layer.bias"],
        "on": sd["transformer.out_norm.scale"],
        "h1": sd["head.projection.hidden_layer.weight"], "h2": sd["head.projection.output_layer.weight"], "hr": sd["head.projection.residual_layer.weight"],
    }
    for i, kind in enumerate(SCHED):
        p = f"transformer.layers.{i}."
        P[f"l{i}_n0"] = sd[p + "attn_norms.0.scale"]
        P[f"l{i}_qkvg"] = torch.cat([reorder_qkv(sd[p + "attns.0.wQKV.weight"]), sd[p + "attns.0._gate.weight"]], 0)
        P[f"l{i}_o0"] = sd[p + "attns.0.wO.weight"]
        P[f"l{i}_nm"] = sd[p + "mlp_norm.scale"]
        P[f"l{i}_w12"] = sd[p + "mlp.w12.weight"]
        P[f"l{i}_w3"] = sd[p + "mlp.w3.weight"]
        if kind == "time_variate":
            P[f"l{i}_n1"] = sd[p + "attn_norms.1.scale"]
            P[f"l{i}_vg"] = torch.cat([reorder_qkv(sd[p + "attns.1.wQKV.weight"])[2 * D:], sd[p + "attns.1._gate.weight"]], 0)
            P[f"l{i}_o1"] = sd[p + "attns.1.wO.weight"]
    return {k: v.contiguous().clone() for k, v in P.items()}


def tables(device="cuda"):
    inv = torch.zeros(HD // 2, dtype=torch.float32, device=device)
    inv[:16] = 1.0 / (10000.0 ** (torch.arange(0, 32, 2, dtype=torch.float32, device=device) / 32))
    fr = torch.arange(33, dtype=torch.float32, device=device)[:, None] * inv[None]
    emb = torch.cat([fr, fr], -1)
    wd = [0.0] * 21
    for i in range(11, 21): wd[i] = LEVELS[i] - LEVELS[i - 1]
    for i in range(10): wd[i] = LEVELS[i + 1] - LEVELS[i]
    return emb.cos(), emb.sin(), 3.0 * torch.tensor(wd, device=device), torch.tensor(LEVELS, device=device)


def rms(x, w): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w


def forward_loss(P, ctx, y, valid, cdt=torch.float32, aux=0.5, fast=False):
    """ctx (B,CTX) f32, y (B,32) f32 next-32 log returns, valid (B,32) {0,1}. Returns (loss, qdenorm (B,32,21))."""
    cos_t, sin_t, qw, levels = tables(ctx.device)
    B, T = ctx.shape
    L = T // PS; S = L + 1
    if fast:
        Pc = {k: v.to(cdt) for k, v in P.items() if v.dim() == 2}
    def lin(x, w):
        if fast:
            return (x.to(cdt) @ w.t()).float()
        return (x.to(cdt) @ w.to(cdt).t()).float()
    def W(key):
        return Pc[key] if fast else P[key]
    with torch.no_grad():
        x64 = ctx.double(); cw = torch.arange(1, T + 1, device=ctx.device, dtype=torch.float64)
        means = x64.cumsum(-1) / cw
        shifted = torch.cat([torch.zeros_like(means[:, :1]), means[:, :-1]], -1)
        m2 = ((x64 - shifted) * (x64 - means)).cumsum(-1)
        scales = (m2 / (cw - 1).clamp(min=1.0)).sqrt().clamp(min=1e-6)
        loc = means[:, PS - 1::PS].float(); sc = scales[:, PS - 1::PS].float()
        z = ((ctx.view(B, L, PS) - loc[..., None]) / sc[..., None].clamp_min(torch.finfo(torch.float32).eps)).asinh()
    e = lin(F.silu(lin(z, W("e1")) + P["e1b"]), W("e2")) + P["e2b"] + lin(z, W("er")) + P["erb"]
    x = torch.cat([e, P["fq"].expand(B, 1, D)], 1)
    cos, sin = cos_t[None, :S, None, :], sin_t[None, :S, None, :]
    def rot(v):
        v1, v2 = v[..., :HD // 2], v[..., HD // 2:]
        return v * cos + torch.cat([-v2, v1], -1) * sin
    for i, kind in enumerate(SCHED):
        h = rms(x, P[f"l{i}_n0"])
        qkvg = lin(h, W(f"l{i}_qkvg"))
        qkv = qkvg[..., :3 * D].view(B, S, 3, NH, HD)
        q, k, v = rot(qkv[:, :, 0]), rot(qkv[:, :, 1]), qkv[:, :, 2]
        if fast:
            o = F.scaled_dot_product_attention(q.transpose(1, 2).to(cdt), k.transpose(1, 2).to(cdt), v.transpose(1, 2).to(cdt), is_causal=True).transpose(1, 2).reshape(B, S, D).float()
        else:
            att = torch.einsum("bshd,bthd->bhst", q, k) * 0.125
            att = att.masked_fill(~torch.tril(torch.ones(S, S, dtype=torch.bool, device=ctx.device)), float("-inf")).softmax(-1)
            o = torch.einsum("bhst,bthd->bshd", att, v).reshape(B, S, D)
        o = o * qkvg[..., 3 * D:].sigmoid()
        x = x + lin(o, W(f"l{i}_o0"))
        if kind == "time_variate":
            h = rms(x, P[f"l{i}_n1"])
            vg = lin(h, W(f"l{i}_vg"))
            x = x + lin(vg[..., :D] * vg[..., D:].sigmoid(), W(f"l{i}_o1"))
        h = rms(x, P[f"l{i}_nm"])
        gu = lin(h, W(f"l{i}_w12"))
        x = x + lin(gu[..., :FF] * F.silu(gu[..., FF:]), W(f"l{i}_w3"))
    d = rms(x[:, -1], P["on"])
    raw = (lin(F.silu(lin(d, W("h1"))), W("h2")) + lin(d, W("hr"))).view(B, 32, 21)
    inc = qw * F.softplus(raw)
    med = raw[..., 10:11]
    up = inc[..., 11:].cumsum(-1)
    dn = inc[..., :10].flip(-1).cumsum(-1).flip(-1)
    qn = torch.cat([med - dn, med, med + up], -1)
    q = qn.sinh() * sc[:, -1, None, None] + loc[:, -1, None, None]
    err = y[..., None] - q
    pin = torch.maximum(levels * err, (levels - 1) * err).mean(-1) / LOSS_SCALE     # (B,32)
    w = torch.full((32,), aux / 31, device=ctx.device); w[0] = 1.0
    wv = w[None] * valid
    loss = (pin * wv).sum() / wv.sum()
    return loss, q


def make_batch(closes_by_sym, rng, B, T, device="cuda"):
    """Random training windows: ctx = T log returns, y = next 32 log returns (NaN beyond data -> invalid)."""
    names = list(closes_by_sym)
    Z = np.zeros((B, T), np.float32); Y = np.zeros((B, 32), np.float32); V = np.zeros((B, 32), np.float32)
    for i in range(B):
        c = closes_by_sym[names[rng.integers(len(names))]]
        t = int(rng.integers(T + 1, len(c) - 1))
        lr = np.diff(np.log(c[t - T - 1:t]))
        Z[i] = lr
        fut = np.diff(np.log(c[t - 1:min(t + 32, len(c))]))
        Y[i, :len(fut)] = fut; V[i, :len(fut)] = 1
    return torch.tensor(Z, device=device), torch.tensor(Y, device=device), torch.tensor(V, device=device)
