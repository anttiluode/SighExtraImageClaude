from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np
import torch
from PIL import Image

from .extraction import extract_boundary_signal
from .inversion import tv_invert
from .scene import BoundaryRegion
from .transport import CornerTransport, TransportConfig

WEAK_EVIDENCE_WARNING = (
    "No evidence that this selected boundary strongly constrains the unseen region under the current model."
)


@dataclass(frozen=True)
class PhotoConfig:
    edge: Literal["left", "right", "top", "bottom"] = "right"
    region_fraction: tuple[float, float, float, float] = (0.55, 0.60, 0.98, 0.92)
    max_side: int = 1024
    smooth_sigma: float = 1.5
    n_measure: int = 64


@dataclass
class PreparedPhoto:
    image_chw: torch.Tensor
    region: BoundaryRegion
    overlay_hwc: np.ndarray
    original_size: tuple[int, int]
    canonical_edge: str = "right"


@dataclass
class PhotoInspection:
    overlay_hwc: np.ndarray
    profile: np.ndarray
    derivative_profile: np.ndarray
    chromatic_profile: np.ndarray
    signal_strength: float
    condition_number: float
    tv_profile: np.ndarray | None
    warnings: list[str]
    likelihood_mode: str
    evidence_note: str


def _as_rgb_uint8(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (np.clip(arr, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _resize_keep_aspect(arr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if max(h, w) <= max_side:
        return arr
    scale = max_side / float(max(h, w))
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    return np.asarray(Image.fromarray(arr).resize((nw, nh), Image.Resampling.LANCZOS))


def _canonicalize(arr: np.ndarray, mask: np.ndarray, edge: str):
    if edge == "right":
        return arr, mask
    if edge == "left":
        return np.flip(arr, axis=1).copy(), np.flip(mask, axis=1).copy()
    if edge == "top":
        return np.rot90(arr, k=-1).copy(), np.rot90(mask, k=-1).copy()
    if edge == "bottom":
        return np.rot90(arr, k=1).copy(), np.rot90(mask, k=1).copy()
    raise ValueError(f"unknown edge: {edge}")


def prepare_photo(image, config: PhotoConfig) -> PreparedPhoto:
    arr0 = _as_rgb_uint8(image)
    oh, ow = arr0.shape[:2]
    arr = _resize_keep_aspect(arr0, config.max_side)
    h, w = arr.shape[:2]
    x0f, y0f, x1f, y1f = config.region_fraction
    if not (0.0 <= x0f < x1f <= 1.0 and 0.0 <= y0f < y1f <= 1.0):
        raise ValueError("region_fraction must satisfy 0<=x0<x1<=1 and 0<=y0<y1<=1")
    x0, x1 = int(round(x0f * w)), int(round(x1f * w))
    y0, y1 = int(round(y0f * h)), int(round(y1f * h))
    x0, x1 = min(x0, w - 1), max(min(x1, w), min(x0 + 1, w))
    y0, y1 = min(y0, h - 1), max(min(y1, h), min(y0 + 1, h))
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    carr, cmask = _canonicalize(arr, mask, config.edge)
    ys, xs = np.where(cmask)
    if len(xs) == 0:
        raise ValueError("selected region is empty after edge transform")
    region = BoundaryRegion(int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
    overlay = carr.copy()
    overlay[region.y0:region.y1, region.x0] = [255, 60, 60]
    overlay[region.y0:region.y1, region.x1 - 1] = [255, 60, 60]
    overlay[region.y0, region.x0:region.x1] = [255, 60, 60]
    overlay[region.y1 - 1, region.x0:region.x1] = [255, 60, 60]
    chw = torch.from_numpy(carr.astype(np.float32) / 255.0).permute(2, 0, 1)
    return PreparedPhoto(chw, region, overlay, (oh, ow))


def inspect_photo(image, config: PhotoConfig, likelihood_mode: str = "affine") -> PhotoInspection:
    prepared = prepare_photo(image, config)
    profile_t = extract_boundary_signal(
        prepared.image_chw,
        prepared.region,
        n_measure=config.n_measure,
        smooth_sigma=config.smooth_sigma,
    )
    derivative = torch.zeros_like(profile_t)
    if profile_t.shape[0] > 2:
        derivative[1:-1] = 0.5 * (profile_t[2:] - profile_t[:-2])
        derivative[0] = derivative[1]
        derivative[-1] = derivative[-2]
    intensity = profile_t.mean(dim=1)
    mean_abs = float(profile_t.abs().mean())
    signal_strength = float(torch.std(intensity) / (torch.mean(intensity.abs()) + 1e-8)) if profile_t.numel() else 0.0
    transport = CornerTransport(TransportConfig(n_measure=config.n_measure, n_angle=config.n_measure, ambient=0.0, gain=1.0))
    tv = None
    if mean_abs > 1e-7:
        tv_t = tv_invert(transport, profile_t, lambda_tv=0.02, iters=120, lr=0.06, residual_mode=likelihood_mode)
        tv = tv_t.detach().cpu().numpy()
    warnings: list[str] = []
    if mean_abs < 2e-4 or signal_strength < 0.03:
        warnings.append(WEAK_EVIDENCE_WARNING)
    if transport.condition_number() > 1e8:
        warnings.append("The selected corner model is strongly ill-conditioned; inverse structure is highly uncertain.")
    chroma = profile_t / profile_t.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return PhotoInspection(
        overlay_hwc=prepared.overlay_hwc,
        profile=profile_t.detach().cpu().numpy(),
        derivative_profile=derivative.detach().cpu().numpy(),
        chromatic_profile=chroma.detach().cpu().numpy(),
        signal_strength=signal_strength,
        condition_number=transport.condition_number(),
        tv_profile=tv,
        warnings=warnings,
        likelihood_mode=likelihood_mode,
        evidence_note=(
            "This panel reports extracted boundary evidence and a model-dependent physics-only inverse. "
            "It does not reveal a unique hidden scene."
        ),
    )
