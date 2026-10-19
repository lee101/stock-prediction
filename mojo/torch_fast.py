import math, numpy as np, torch
import torch.nn.functional as F

H, D, NH, HD, FF, PS, L, S = 768, 768, 12, 64, 2048, 32, 16, 17
SCHED = ["time", "time", "time", "time_variate"] * 6


def prep_weights(model, wdtype=torch.float32, device="cuda"):
    sd = {k: v.detach().to(device) for k, v in model.state_dict().items()}
    def reorder_qkv(w):  # rows (c, d, h) -> (c, h, d)
        return w.reshape(3, HD, NH, H).permute(0, 2, 1, 3).reshape(3 * D, H)
    W = dict(
        fq=sd["representation.forecast_query"].float(),
        e1=sd["representation.patch_proj.hidden_layer.weight"][:, :PS].to(wdtype), e1b=sd["representation.patch_proj.hidden_layer.bias"].float(),
        e2=sd["representation.patch_proj.output_layer.weight"].to(wdtype), e2b=sd["representation.patch_proj.output_layer.bias"].float(),
        er=sd["representation.patch_proj.residual_layer.weight"][:, :PS].to(wdtype), erb=sd["representation.patch_proj.residual_layer.bias"].float(),
        on=sd["transformer.out_norm.scale"].float(),
        h1=sd["head.projection.hidden_layer.weight"].to(wdtype), h2=sd["head.projection.output_layer.weight"][:21].to(wdtype), hr=sd["head.projection.residual_layer.weight"][:21].to(wdtype),
        layers=[],
    )
    for i, kind in enumerate(SCHED):
        p = f"transformer.layers.{i}."
        ly = dict(n0=sd[p + "attn_norms.0.scale"].float(), qkv0=reorder_qkv(sd[p + "attns.0.wQKV.weight"]).to(wdtype), g0=sd[p + "attns.0._gate.weight"].to(wdtype), o0=sd[p + "attns.0.wO.weight"].to(wdtype),
                  nm=sd[p + "mlp_norm.scale"].float(), w12=sd[p + "mlp.w12.weight"].to(wdtype), w3=sd[p + "mlp.w3.weight"].to(wdtype), variate=kind == "time_variate")
        if kind == "time_variate":
            ly.update(n1=sd[p + "attn_norms.1.scale"].float(), v1=reorder_qkv(sd[p + "attns.1.wQKV.weight"])[2 * D:].to(wdtype), g1=sd[p + "attns.1._gate.weight"].to(wdtype), o1=sd[p + "attns.1.wO.weight"].to(wdtype))
        W["layers"].append(ly)
    inv = torch.zeros(HD // 2, dtype=torch.float32, device=device)
    inv[:16] = 1.0 / (10000.0 ** (torch.arange(0, 32, 2, dtype=torch.float32, device=device) / 32))
    fr = torch.arange(S, dtype=torch.float32, device=device)[:, None] * inv[None]
    emb = torch.cat([fr, fr], -1)
    W["cos"], W["sin"] = emb.cos(), emb.sin()
    lv = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    wd = [0.0] * 21
    for i in range(11, 21): wd[i] = lv[i] - lv[i - 1]
    for i in range(10): wd[i] = lv[i + 1] - lv[i]
    W["qw"] = 3.0 * torch.tensor(wd, device=device)
    return W


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w


def lin(x, w):
    return (x.to(w.dtype) @ w.t()).float()


def rot(x, cos, sin):
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


def forward(W, ctx):
    """ctx (B,512) float32 -> normalized-space quantiles (B,21) denormalized, float32."""
    B = ctx.shape[0]
    x64 = ctx.double()
    cw = torch.arange(1, ctx.shape[1] + 1, device=ctx.device, dtype=torch.float64)
    means = x64.cumsum(-1) / cw
    shifted = torch.cat([torch.zeros_like(means[:, :1]), means[:, :-1]], -1)
    m2 = ((x64 - shifted) * (x64 - means)).cumsum(-1)
    var = m2 / (cw - 1).clamp(min=1.0)
    scales = var.sqrt().clamp(min=1e-6)
    loc = means[:, PS - 1::PS].float(); sc = scales[:, PS - 1::PS].float()             # (B,16)
    z = (ctx.view(B, L, PS) - loc[..., None]) / sc[..., None].clamp_min(torch.finfo(torch.float32).eps)
    z = z.asinh()
    e = lin(F.silu(lin(z, W["e1"]) + W["e1b"]), W["e2"]) + W["e2b"] + lin(z, W["er"]) + W["erb"]
    x = torch.cat([e, W["fq"].expand(B, 1, D)], 1)                                       # (B,17,768)
    cos, sin = W["cos"][None, :, None, :], W["sin"][None, :, None, :]
    for ly in W["layers"]:
        h = rms(x, ly["n0"])
        qkv = lin(h, ly["qkv0"]).view(B, S, 3, NH, HD)
        q, k, v = rot(qkv[:, :, 0], cos, sin), rot(qkv[:, :, 1], cos, sin), qkv[:, :, 2]
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2).reshape(B, S, D)
        x = x + lin(o * lin(h, ly["g0"]).sigmoid(), ly["o0"])
        if ly["variate"]:
            h = rms(x, ly["n1"])
            x = x + lin(lin(h, ly["v1"]) * lin(h, ly["g1"]).sigmoid(), ly["o1"])
        h = rms(x, ly["nm"])
        gu = lin(h, ly["w12"]); g, u = gu[..., :FF], gu[..., FF:]
        x = x + lin(g * F.silu(u), ly["w3"])
    d = rms(x[:, -1], W["on"])
    raw = lin(F.silu(lin(d, W["h1"])), W["h2"]) + lin(d, W["hr"])                       # (B,21)
    inc = W["qw"] * F.softplus(raw)
    med = raw[:, 10:11]
    up = inc[:, 11:].cumsum(-1)
    dn = inc[:, :10].flip(-1).cumsum(-1).flip(-1)
    qn = torch.cat([med - dn, med, med + up], -1)
    return qn.sinh() * sc[:, -1:] + loc[:, -1:]


def forecast(W, ctx, sa=False):
    q = forward(W, ctx)
    if sa:
        q = 0.5 * (q + (-forward(W, -ctx)).flip(-1))
    return q
