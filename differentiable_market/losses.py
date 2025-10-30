from __future__ import annotations

import torch
from torch import Tensor


def dirichlet_kl(alpha: Tensor, beta: Tensor) -> Tensor:
    """
    Kullback-Leibler divergence KL(alpha || beta) for Dirichlet parameters.
    """
    if alpha.shape != beta.shape:
        raise ValueError("alpha and beta must share the same shape")
    sum_alpha = alpha.sum(dim=-1)
    sum_beta = beta.sum(dim=-1)
    term1 = torch.lgamma(sum_alpha) - torch.lgamma(sum_beta)
    term2 = torch.lgamma(beta).sum(dim=-1) - torch.lgamma(alpha).sum(dim=-1)
    term3 = ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(sum_alpha).unsqueeze(-1))).sum(dim=-1)
    return term1 + term2 + term3

