from __future__ import annotations

from dataclasses import dataclass
import torch

from .latent import PhysicalLatent, PRIOR_RANGES
from .likelihood import physics_residual


_CONT_LO = torch.tensor([
    PRIOR_RANGES["theta"][0], PRIOR_RANGES["width"][0], PRIOR_RANGES["height"][0],
    PRIOR_RANGES["rgb"][0], PRIOR_RANGES["rgb"][0], PRIOR_RANGES["rgb"][0],
    PRIOR_RANGES["brightness"][0],
], dtype=torch.float32)
_CONT_HI = torch.tensor([
    PRIOR_RANGES["theta"][1], PRIOR_RANGES["width"][1], PRIOR_RANGES["height"][1],
    PRIOR_RANGES["rgb"][1], PRIOR_RANGES["rgb"][1], PRIOR_RANGES["rgb"][1],
    PRIOR_RANGES["brightness"][1],
], dtype=torch.float32)


@dataclass
class GradientStepResult:
    component_alignment: torch.Tensor
    error_before: float
    error_after: float
    error_reduction: float
    gradient_norm: float
    finite: bool
    loss_before: float
    loss_after: float


@dataclass
class Gate1Result:
    correct: GradientStepResult
    wrong: GradientStepResult
    no_occluder: GradientStepResult
    random: GradientStepResult


def _bounds_like(v: torch.Tensor):
    return _CONT_LO.to(v), _CONT_HI.to(v)


def continuous_vector(latent: PhysicalLatent) -> torch.Tensor:
    if latent.batch_size != 1:
        raise ValueError("Gate 1 currently operates on one latent at a time")
    raw = torch.cat([
        latent.theta[:1], latent.width[:1], latent.height[:1],
        latent.rgb[0], latent.brightness[:1]
    ])
    lo, hi = _bounds_like(raw)
    return (raw - lo) / (hi - lo)


def latent_from_continuous(v: torch.Tensor, shape_code: int) -> PhysicalLatent:
    lo, hi = _bounds_like(v)
    raw = lo + v * (hi - lo)
    return PhysicalLatent(
        theta=raw[0:1], width=raw[1:2], height=raw[2:3],
        rgb=raw[3:6].unsqueeze(0), brightness=raw[6:7],
        shape_code=torch.tensor([shape_code], dtype=torch.long, device=v.device),
    )


def boundary_loss(latent: PhysicalLatent, observed_y: torch.Tensor, transport, scene, residual_mode: str = "l2") -> torch.Tensor:
    pred = transport.forward_latent(latent, scene)[0]
    if observed_y.ndim == 3:
        observed_y = observed_y[0]
    return physics_residual(observed_y, pred, residual_mode)


def _evaluate_direction(
    v_true: torch.Tensor,
    v_init: torch.Tensor,
    observed_y: torch.Tensor,
    transport,
    scene,
    residual_mode: str,
    step_size: float,
    shape_code: int,
    *,
    forced_direction: torch.Tensor | None = None,
) -> GradientStepResult:
    v = v_init.detach().clone().requires_grad_(True)
    loss = boundary_loss(latent_from_continuous(v, shape_code), observed_y, transport, scene, residual_mode)
    grad = torch.autograd.grad(loss, v)[0]
    grad_norm = torch.linalg.vector_norm(grad)
    if forced_direction is None:
        direction = -grad
    else:
        direction = forced_direction.to(v)
    dir_norm = torch.linalg.vector_norm(direction)
    finite = bool(torch.isfinite(grad).all() and torch.isfinite(loss) and torch.isfinite(dir_norm))
    if float(dir_norm) > 1e-12:
        unit_direction = direction / dir_norm
    else:
        unit_direction = torch.zeros_like(direction)
    truth_delta = v_true - v_init
    component_alignment = unit_direction * truth_delta
    error_before = torch.linalg.vector_norm(truth_delta)
    v_after = (v_init + float(step_size) * unit_direction).clamp(0.0, 1.0)
    error_after = torch.linalg.vector_norm(v_true - v_after)
    loss_after = boundary_loss(latent_from_continuous(v_after, shape_code), observed_y, transport, scene, residual_mode)
    return GradientStepResult(
        component_alignment=component_alignment.detach(),
        error_before=float(error_before), error_after=float(error_after),
        error_reduction=float(error_before - error_after),
        gradient_norm=float(grad_norm), finite=finite,
        loss_before=float(loss.detach()), loss_after=float(loss_after.detach()),
    )


def evaluate_gate1(
    z_true: PhysicalLatent,
    z_init: PhysicalLatent,
    observed_y: torch.Tensor,
    correct_transport,
    wrong_transport,
    no_occluder_transport,
    scene,
    *,
    residual_mode: str = "l2",
    step_size: float = 0.1,
    random_generator: torch.Generator | None = None,
) -> Gate1Result:
    v_true = continuous_vector(z_true).detach()
    v_init = continuous_vector(z_init).detach()
    shape_code = int(z_init.shape_code[0])
    correct = _evaluate_direction(v_true, v_init, observed_y, correct_transport, scene, residual_mode, step_size, shape_code)
    wrong = _evaluate_direction(v_true, v_init, observed_y, wrong_transport, scene, residual_mode, step_size, shape_code)
    no_occ = _evaluate_direction(v_true, v_init, observed_y, no_occluder_transport, scene, residual_mode, step_size, shape_code)
    if random_generator is None:
        random_generator = torch.Generator().manual_seed(0)
    rnd = torch.randn(v_init.shape, generator=random_generator, dtype=v_init.dtype, device=v_init.device)
    random = _evaluate_direction(
        v_true, v_init, observed_y, correct_transport, scene, residual_mode, step_size, shape_code,
        forced_direction=rnd,
    )
    return Gate1Result(correct=correct, wrong=wrong, no_occluder=no_occ, random=random)
