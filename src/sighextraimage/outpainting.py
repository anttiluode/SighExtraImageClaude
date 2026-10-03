"""Photo geometry and optional coarse light constraints for speculative outpainting."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from PIL import Image
import torch

from .extraction import extract_boundary_signal, srgb_to_linear
from .photo import PhotoConfig, _as_rgb_uint8, _canonicalize, _resize_keep_aspect, prepare_photo
from .transport import CornerTransport, TransportConfig


DEFAULT_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"


@dataclass(frozen=True)
class OutpaintConfig:
    edge: str = "right"
    extension_fraction: float = 0.45
    max_side: int = 512
    prompt: str = "a photograph of the room continuing naturally, consistent lighting and perspective"
    model_id: str = DEFAULT_MODEL
    steps: int = 30
    seed: int = 42
    text_guidance: float = 7.0
    light_guidance: bool = False
    physics_strength: float = 0.08

    def __post_init__(self):
        if self.edge not in {"right", "left", "top", "bottom"}:
            raise ValueError("edge must be right, left, top, or bottom")
        if not math.isfinite(self.extension_fraction) or not 0 < self.extension_fraction <= 2:
            raise ValueError("extension fraction must be greater than zero and at most 2")
        if not 64 <= self.max_side <= 1536:
            raise ValueError("working maximum side must be between 64 and 1536")
        if not 1 <= self.steps <= 100:
            raise ValueError("sampling steps must be between 1 and 100")
        if not math.isfinite(self.physics_strength) or not 0 <= self.physics_strength <= 0.25:
            raise ValueError("light guidance strength must be between 0 and 0.25")
        if not math.isfinite(self.text_guidance) or not 1 <= self.text_guidance <= 20:
            raise ValueError("text guidance must be between 1 and 20")
        if not self.prompt.strip() or not self.model_id.strip():
            raise ValueError("prompt and model ID must not be empty")


@dataclass
class OutpaintCanvas:
    canvas: torch.Tensor
    mask: torch.Tensor
    visible_width: int
    visible_height: int
    extension_width: int
    original_rgb: np.ndarray
    edge: str
    target_extension_width: int


def prepare_canvas(image, config: OutpaintConfig) -> OutpaintCanvas:
    original = _as_rgb_uint8(image)
    canonical, _ = _canonicalize(original, np.zeros(original.shape[:2], dtype=bool), config.edge)
    working = _resize_keep_aspect(canonical, config.max_side)
    h, w = working.shape[:2]
    extra = max(1, round(w * config.extension_fraction))
    # SD 1.5's VAE and UNet together prefer a multiple of 64. Padding is
    # generated but cropped away; resizing never stretches the visible photo.
    ph = math.ceil(h / 64) * 64
    pw = math.ceil((w + extra) / 64) * 64
    canvas = torch.full((3, ph, pw), 0.5)
    canvas[:, :h, :w] = torch.from_numpy(working.astype(np.float32) / 255).permute(2, 0, 1)
    mask = torch.ones((1, ph, pw))
    mask[:, :h, :w] = 0
    return OutpaintCanvas(
        canvas, mask, w, h, extra, original, config.edge,
        max(1, round(canonical.shape[1] * config.extension_fraction)),
    )


def restore_outpaint(generated: torch.Tensor, prepared: OutpaintCanvas) -> np.ndarray:
    """Keep only generated pixels outside the original FoV, then undo orientation."""
    if generated.shape != prepared.canvas.shape or not torch.isfinite(generated).all():
        raise ValueError("generator returned an invalid image tensor")
    canonical, _ = _canonicalize(
        prepared.original_rgb, np.zeros(prepared.original_rgb.shape[:2], dtype=bool), prepared.edge,
    )
    hidden = generated[:, :prepared.visible_height,
                       prepared.visible_width:prepared.visible_width + prepared.extension_width]
    hidden_u8 = (hidden.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8)
    hidden_u8 = np.asarray(Image.fromarray(hidden_u8).resize(
        (prepared.target_extension_width, canonical.shape[0]), Image.Resampling.LANCZOS,
    ))
    result = np.concatenate([canonical, hidden_u8], axis=1)
    if prepared.edge == "left":
        result = np.flip(result, axis=1)
    elif prepared.edge == "top":
        result = np.rot90(result, k=1)
    elif prepared.edge == "bottom":
        result = np.rot90(result, k=-1)
    return result.copy()


@dataclass
class CoarseLightConstraint:
    transport: CornerTransport
    measured: torch.Tensor
    basis: torch.Tensor
    allowed: bool
    reason: str

    @property
    def retained_modes(self) -> int:
        return self.basis.shape[1]

    @classmethod
    def from_measurement(cls, measured: torch.Tensor, transport: CornerTransport,
                         *, cutoff: float = 0.01) -> CoarseLightConstraint:
        y = measured.detach().float().cpu()
        if y.shape != (transport.config.n_measure, 3):
            raise ValueError("light measurement must be [n_measure,3]")
        empty = torch.zeros((y.shape[0], 0))
        if not torch.isfinite(y).all():
            return cls(transport, y, empty, False, "The boundary measurement is not finite.")
        centered = y - y.mean(0, keepdim=True)
        contrast = centered.square().mean().sqrt() / y.abs().mean().clamp_min(1e-8)
        if float(y.abs().mean()) < 2e-4 or float(contrast) < 0.03:
            return cls(transport, y, empty, False, "The selected boundary is too weak or flat for light guidance.")
        kernel = transport.kernel.detach().double().cpu()
        kernel = kernel - kernel.mean(0, keepdim=True)
        u, s, _ = torch.linalg.svd(kernel, full_matrices=False)
        keep = (s >= s[0] * cutoff) & (s > 1e-7)
        basis = u[:, keep].float()
        if basis.shape[1] < 2:
            return cls(transport, y, basis, False, "The transport has too few distinguishable coarse modes.")
        if float((basis.T @ centered).square().sum()) < 1e-10:
            return cls(transport, y, basis, False, "The measurement has no usable coarse transport component.")
        return cls(transport, y, basis, True,
                   "Experimental light guidance uses coarse modes of an assumed corner model.")

    def residual(self, predicted: torch.Tensor) -> torch.Tensor:
        if not self.allowed:
            raise ValueError(self.reason)
        if predicted.shape != self.measured.shape:
            raise ValueError("predicted boundary shape does not match the measurement")
        basis = self.basis.to(predicted)
        target = self.measured.to(predicted)
        yc = basis.T @ (target - target.mean(0, keepdim=True))
        hc = basis.T @ (predicted - predicted.mean(0, keepdim=True))
        # Exposure is positive. Negative scaling would let an inverted shadow
        # count as a match. Offset/scale remain nuisance parameters, not evidence.
        gain = ((yc * hc).sum() / hc.square().sum().clamp_min(1e-12)).clamp_min(0)
        return (yc - gain * hc).square().sum() / yc.square().sum().clamp_min(1e-12)

    def loss(self, hidden_srgb: torch.Tensor) -> torch.Tensor:
        predicted = self.transport.forward_hidden(srgb_to_linear(hidden_srgb))[0]
        return self.residual(predicted)


@dataclass
class OutpaintComparison:
    prior: np.ndarray
    guided: np.ndarray
    guidance_status: str
    warnings: list[str]
    diagnostics: dict


def generate_outpaintings(image, photo_config: PhotoConfig, config: OutpaintConfig, *,
                         engine=None, progress=None) -> OutpaintComparison:
    """Generate the baseline first; optional evidence may modify only the comparison."""
    from dataclasses import replace
    if photo_config.edge != config.edge:
        raise ValueError("measurement edge and extension edge must match")
    prepared = prepare_canvas(image, config)
    if engine is None:
        from .diffusion import get_engine
        if progress is not None:
            progress(0, "Loading image model; first use downloads the weights")
        engine = get_engine(config.model_id)
    def phase_progress(start, span):
        return None if progress is None else lambda f, desc: progress(start + span*f, desc)
    prior_tensor, prior_run = engine.generate_with_diagnostics(prepared, config,
        progress=phase_progress(0, 0.5 if config.light_guidance else 1))
    prior = restore_outpaint(prior_tensor, prepared)
    guided = prior.copy()
    status = "not_requested"
    warnings = []
    diagnostics = {"model": config.model_id, "seed": config.seed, "steps": config.steps,
                   "extension_edge": config.edge, "output_shape": list(prior.shape),
                   "interpretation": "Generated hypotheses; outside-FoV details are not measured ground truth."}
    if "device" in prior_run:
        diagnostics["device"] = prior_run["device"]
    if config.light_guidance:
        try:
            photo = prepare_photo(image, replace(photo_config, max_side=config.max_side))
            y = extract_boundary_signal(photo.image_chw, photo.region,
                n_measure=photo_config.n_measure, smooth_sigma=photo_config.smooth_sigma)
            transport = CornerTransport(TransportConfig(
                n_measure=photo_config.n_measure, n_angle=photo_config.n_measure))
            constraint = CoarseLightConstraint.from_measurement(y, transport)
            diagnostics["retained_light_modes"] = constraint.retained_modes
            diagnostics["light_note"] = constraint.reason
            if constraint.allowed and config.physics_strength > 0:
                guided_tensor, guided_run = engine.generate_with_diagnostics(
                    prepared, config, constraint=constraint, progress=phase_progress(0.5, 0.5))
                guided = restore_outpaint(guided_tensor, prepared)
                status = "applied" if guided_run.get("guidance_steps", 0) else "skipped"
                diagnostics["guidance_steps"] = guided_run.get("guidance_steps", 0)
                for label, tensor in (("prior", prior_tensor), ("guided", guided_tensor)):
                    hidden = tensor[:, :prepared.visible_height,
                                    prepared.visible_width:prepared.visible_width + prepared.extension_width]
                    diagnostics[label + "_coarse_residual"] = float(constraint.loss(hidden))
                if status == "skipped":
                    warnings.append(guided_run.get("guidance_disabled_reason",
                        "No usable light-gradient update was available; the comparison is prior-only."))
            else:
                status = "skipped"
                warnings.append(constraint.reason if not constraint.allowed else "Light guidance strength is zero.")
        except Exception as exc:
            status = "failed"
            warnings.append(f"Light guidance failed; the prior extension is retained: {exc}")
    if progress is not None:
        progress(1, "Image extension complete")
    return OutpaintComparison(prior, guided, status, warnings, diagnostics)
