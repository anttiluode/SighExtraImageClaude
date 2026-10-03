from __future__ import annotations

from typing import Literal
import torch


ResidualMode = Literal["l2", "affine", "affine_per_channel"]


def physics_residual(y: torch.Tensor, yhat: torch.Tensor, mode: ResidualMode = "l2") -> torch.Tensor:
    if y.shape != yhat.shape:
        raise ValueError(f"shape mismatch: {tuple(y.shape)} vs {tuple(yhat.shape)}")
    if mode == "l2":
        return torch.mean((y - yhat) ** 2)
    yc = y - y.mean(dim=-2, keepdim=True)
    hc = yhat - yhat.mean(dim=-2, keepdim=True)
    eps = torch.as_tensor(1e-12, dtype=y.dtype, device=y.device)
    if mode == "affine":
        a = (yc * hc).sum() / (hc * hc).sum().clamp_min(eps)
        resid = yc - a * hc
    elif mode == "affine_per_channel":
        a = (yc * hc).sum(dim=-2, keepdim=True) / (hc * hc).sum(dim=-2, keepdim=True).clamp_min(eps)
        resid = yc - a * hc
    else:
        raise ValueError(f"unknown residual mode: {mode}")
    return (resid * resid).sum() / (yc * yc).sum().clamp_min(eps)
