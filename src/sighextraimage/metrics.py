from __future__ import annotations

import math
import torch


def scalar_error(value: float, truth: float) -> float:
    return abs(float(value) - float(truth))


def rgb_error(value: torch.Tensor, truth: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach().float() - truth.detach().float()))


def contraction_ratio(posterior_std: float, prior_std: float) -> float:
    denom = max(abs(float(prior_std)), 1e-12)
    return float(abs(float(posterior_std)) / denom)


def finite_float(value: float, fallback: float = 0.0) -> float:
    value = float(value)
    return value if math.isfinite(value) else float(fallback)
