"""Corner geometry as a refutable hypothesis, estimated by a probe -> residue funnel.

Ported in spirit from Varjoluotain's occluder funnel
(https://github.com/anttiluode/Varjoluotain, ``varjoluotain/locate.py``):

* every candidate corner geometry is a hypothesis about the room;
* a hypothesis is refuted when no non-negative hidden radiance can explain the
  boundary profile under it, beyond what the photo's own noise allows;
* a cheap model tests a coarse grid, and only unrefuted cells are refined.

Hypotheses that the data *can* decide are used here. Mirroring the hidden-angle
axis is not one of them: ``WrongCornerTransport`` at theta predicts exactly what
``CornerTransport`` predicts at ``theta_max - theta``. It is a relabeling, so it
is not part of the hypothesis set. The refutable alternatives are:

* ``phi_start, phi_end``: which corner angles the selected strip spans;
* ``penumbra``: how soft the edge's shadow is;
* ``reversed``: the corner edge sits at the other end of the strip;
* background only: no occluder signature at all (a flat kernel is removed
  entirely by the background projection).

Noise comes from the photo itself (split-half of the selected rows), and the
likelihood temperature is the Birge ratio of the best fit, so a model that
cannot reach the noise floor is tempered instead of becoming certain.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
import time

import numpy as np
import torch
from scipy.optimize import nnls
from scipy.stats import chi2

from .extraction import extract_boundary_profile
from .scene import BoundaryRegion
from .transport import CornerTransport, TransportConfig


# ----------------------------------------------------------------- hypotheses --
@dataclass(frozen=True)
class CornerHypothesis:
    phi_start: float
    phi_end: float
    penumbra: float
    reversed: bool = False

    def transport(self, n_measure: int, n_angle: int) -> CornerTransport:
        cfg = TransportConfig(
            n_measure=n_measure, n_angle=n_angle,
            phi_min=float(self.phi_start), phi_max=float(self.phi_end),
            penumbra=float(self.penumbra), ambient=0.0, gain=1.0,
        )
        t = CornerTransport(cfg)
        if self.reversed:
            t.kernel.copy_(torch.flip(t.kernel, dims=[0]))
        return t

    def kernel(self, n_measure: int, n_angle: int) -> np.ndarray:
        """Kernel on the scene's fixed hidden-angle grid (used for parametric scenes)."""
        return self.transport(n_measure, n_angle).kernel.double().numpy()

    def funnel_kernel(self, n_measure: int, n_angle: int) -> np.ndarray:
        """Kernel whose free radiance bins tile this hypothesis's own visible span.

        Radiance at angles below the span lights every measurement equally (it is
        removed with the background); radiance above it reaches no measurement.
        Tiling only the span gives every hypothesis the same number of useful
        bins, so no geometry wins merely by discretisation.
        """
        p = float(self.penumbra)
        lo = max(0.0, float(self.phi_start) - 3.0 * p)
        hi = min(1.30, float(self.phi_end) + 3.0 * p)
        cfg = TransportConfig(n_measure=n_measure, n_angle=n_angle,
                              phi_min=float(self.phi_start), phi_max=float(self.phi_end),
                              penumbra=p, theta_min=lo, theta_max=max(hi, lo + 1e-3),
                              ambient=0.0, gain=1.0)
        k = CornerTransport(cfg).kernel.double().numpy()
        return k[::-1].copy() if self.reversed else k

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_HYPOTHESIS = CornerHypothesis(0.04, 1.25, 0.055, False)


def _vec(h: CornerHypothesis) -> np.ndarray:
    return np.array([h.phi_start, h.phi_end, math.log(h.penumbra)])


def _hyp(v, reversed_: bool) -> CornerHypothesis:
    return CornerHypothesis(float(v[0]), float(v[1]), float(math.exp(v[2])), bool(reversed_))


# ------------------------------------------------------------ background ------
def background_basis(n: int, order: int = 1) -> np.ndarray:
    u = np.linspace(-1.0, 1.0, n)
    cols = [np.ones(n)] + [u ** k for k in range(1, order + 1)]
    q, _ = np.linalg.qr(np.stack(cols, 1))
    return q


def project_out(m: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Remove the unknown room light (constant + ramp) along the measurement axis."""
    return m - q @ (q.T @ m)


# ------------------------------------------------------------------- noise ----
@dataclass
class NoiseEstimate:
    sigma: np.ndarray            # per-channel std of the full-strip profile
    correlation: float           # >= 1: how many samples one independent sample spans
    method: str

    @property
    def effective_fraction(self) -> float:
        return 1.0 / self.correlation


def _integrated_correlation(d: np.ndarray, max_lag: int = 12) -> float:
    """Integrated autocorrelation time of a noise trace, summed until it turns negative."""
    d = d - d.mean(0, keepdims=True)
    var = (d * d).sum(0)
    tau = np.ones(d.shape[1])
    for c in range(d.shape[1]):
        if var[c] <= 0:
            continue
        acc = 1.0
        for lag in range(1, min(max_lag, d.shape[0] - 1)):
            r = float((d[lag:, c] * d[:-lag, c]).sum() / var[c])
            if r <= 0:
                break
            acc += 2.0 * r
        tau[c] = acc
    return float(max(1.0, np.median(tau)))


def split_half_noise(image: torch.Tensor, region: BoundaryRegion, *, n_measure: int,
                     smooth_sigma: float, block: int = 4, order: int = 1) -> NoiseEstimate:
    """Noise of the strip profile from the photo itself.

    Rows are split into two interleaved halves (alternating blocks), each half is
    passed through the same extraction, and half their difference estimates the
    noise of the full-strip mean. Interleaving keeps the vertical illumination
    weight the same in both halves; a constant + ramp is removed from the
    difference so a smooth exposure mismatch is not counted as noise.
    """
    rows = np.arange(region.y0, region.y1)
    if len(rows) < 4 * block:
        block = max(1, len(rows) // 4)
    if len(rows) < 4:
        raise ValueError("boundary region needs at least four rows for a split-half noise estimate")
    a = rows[(np.arange(len(rows)) // block) % 2 == 0]
    b = rows[(np.arange(len(rows)) // block) % 2 == 1]
    pa = extract_boundary_profile(image, region, n_measure=n_measure, smooth_sigma=smooth_sigma, rows=a)
    pb = extract_boundary_profile(image, region, n_measure=n_measure, smooth_sigma=smooth_sigma, rows=b)
    d = project_out((pa - pb).double().numpy(), background_basis(n_measure, order)) / 2.0
    mad = np.median(np.abs(d - np.median(d, 0)), 0) * 1.4826
    # 8-bit quantisation floor for the mean over the pixels behind one profile sample
    n_px = max(1.0, len(rows) * (region.x1 - region.x0) / n_measure)
    floor = (1.0 / 255.0) / math.sqrt(12.0) / math.sqrt(n_px) * 0.5
    sigma = np.maximum(np.maximum(mad, d.std(0)), floor)
    return NoiseEstimate(sigma=sigma, correlation=_integrated_correlation(d), method="split-half interleaved rows")


# ------------------------------------------------------------------- misfit ---
def nn_misfit(kernel: np.ndarray, y: np.ndarray, sigma: np.ndarray, q: np.ndarray) -> float:
    """Chi-square left by the best non-negative hidden radiance (each channel free)."""
    kp = project_out(kernel, q)
    yp = project_out(y, q)
    total = 0.0
    for c in range(y.shape[1]):
        _, rnorm = nnls(kp, yp[:, c], maxiter=50 * kp.shape[1])
        total += (rnorm / sigma[c]) ** 2
    return float(total)


def background_only_chi2(y: np.ndarray, sigma: np.ndarray, q: np.ndarray) -> float:
    yp = project_out(y, q)
    return float(((yp / sigma[None, :]) ** 2).sum())


# ------------------------------------------------------------------- funnel ---
@dataclass(frozen=True)
class FunnelLevel:
    n_measure: int
    n_angle: int
    step: tuple[float, float, float]   # (phi_start, phi_end, log penumbra)


@dataclass(frozen=True)
class FunnelConfig:
    phi_start: tuple[float, float] = (0.0, 0.45)
    phi_end: tuple[float, float] = (0.75, 1.30)
    log_penumbra: tuple[float, float] = (math.log(0.02), math.log(0.16))
    levels: tuple[FunnelLevel, ...] = (
        FunnelLevel(32, 12, (0.05, 0.05, 0.35)),
        FunnelLevel(64, 16, (0.025, 0.025, 0.175)),
    )
    survival_quantile: float = 0.99      # Delta chi2 quantile, k = 3 geometry parameters
    adequacy_birge: float = 4.0          # best reduced chi2 above this: corner model refuted
    refine_if_refuted: float = 0.5       # refine only when a level refuted at least this share
    max_children: int = 6000
    order: int = 1


@dataclass
class FunnelResult:
    best: CornerHypothesis
    survivors: list[CornerHypothesis]
    survivor_chi2: list[float]
    ledger: list[dict]
    best_chi2: float
    dof: int
    birge: float                     # reduced chi2 of the best NNLS fit
    background_chi2: float
    background_delta: float          # background chi2 minus best, in units of the survival threshold
    background_refuted: bool
    reversed_refuted: bool
    model_adequate: bool
    box_refuted_fraction: float      # share of the coarse box the photo rules out
    geometry_decided: bool
    noise: NoiseEstimate
    verdict: str
    status: str                      # "refuted-model" | "no-signal" | "undecided-geometry" | "decided-geometry"

    @property
    def temperature(self) -> float:
        return max(1.0, self.birge)

    @property
    def signal_present(self) -> bool:
        return self.model_adequate and self.background_refuted

    def summary(self) -> dict:
        return {
            "status": self.status, "verdict": self.verdict,
            "best": self.best.as_dict(),
            "n_survivors": len(self.survivors),
            "survivor_ranges": _ranges(self.survivors),
            "box_refuted_fraction": self.box_refuted_fraction,
            "geometry_decided": self.geometry_decided,
            "best_chi2": self.best_chi2, "dof": self.dof, "birge": self.birge,
            "temperature": self.temperature,
            "background_chi2": self.background_chi2,
            "background_delta": self.background_delta,
            "background_refuted": self.background_refuted,
            "reversed_refuted": self.reversed_refuted,
            "model_adequate": self.model_adequate,
            "noise_sigma": self.noise.sigma.tolist(),
            "noise_correlation": self.noise.correlation,
            "ledger": self.ledger,
        }


def _ranges(hs: list[CornerHypothesis]) -> dict:
    if not hs:
        return {}
    a = np.array([[h.phi_start, h.phi_end, h.penumbra] for h in hs])
    return {
        "phi_start": [float(a[:, 0].min()), float(a[:, 0].max())],
        "phi_end": [float(a[:, 1].min()), float(a[:, 1].max())],
        "penumbra": [float(a[:, 2].min()), float(a[:, 2].max())],
        "reversed": sorted({bool(h.reversed) for h in hs}),
    }


def _grid(cfg: FunnelConfig, step) -> np.ndarray:
    axes = [np.arange(lo, hi + 1e-9, st) for (lo, hi), st in
            zip((cfg.phi_start, cfg.phi_end, cfg.log_penumbra), step)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    return np.concatenate([np.c_[g, np.zeros(len(g))], np.c_[g, np.ones(len(g))]], 0)


def _children(parents: np.ndarray, step) -> np.ndarray:
    offs = np.stack(np.meshgrid(*[np.array([-1, 0, 1])] * 3, indexing="ij"), -1).reshape(-1, 3)
    kids = (parents[:, None, :3] + offs[None] * np.asarray(step)).reshape(-1, 3)
    rev = np.repeat(parents[:, 3], len(offs))
    allv = np.concatenate([kids, rev[:, None]], 1)
    key = np.round(allv / np.r_[np.asarray(step) / 4, 1.0]).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return allv[np.sort(idx)]


def _in_box(H: np.ndarray, cfg: FunnelConfig) -> np.ndarray:
    ok = np.ones(len(H), bool)
    for j, (lo, hi) in enumerate((cfg.phi_start, cfg.phi_end, cfg.log_penumbra)):
        ok &= (H[:, j] >= lo - 1e-9) & (H[:, j] <= hi + 1e-9)
    return ok & (H[:, 0] < H[:, 1] - 0.2)


def _resample(y: np.ndarray, n: int) -> np.ndarray:
    if y.shape[0] == n:
        return y
    x0 = np.linspace(0, 1, y.shape[0]); x1 = np.linspace(0, 1, n)
    return np.stack([np.interp(x1, x0, y[:, c]) for c in range(y.shape[1])], 1)


def _level_costs(H, lvl, y_full, noise, order):
    y = _resample(y_full, lvl.n_measure)
    q = background_basis(lvl.n_measure, order)
    cost = np.array([nn_misfit(_hyp(h[:3], h[3]).funnel_kernel(lvl.n_measure, lvl.n_angle), y, noise.sigma, q)
                     for h in H])
    return cost, y, q


def run_funnel(profile: torch.Tensor | np.ndarray, noise: NoiseEstimate,
               config: FunnelConfig = FunnelConfig(), *, truth: CornerHypothesis | None = None,
               log=None) -> FunnelResult:
    """What one boundary profile decides about the corner geometry.

    ``truth`` is only used to report whether the true cell was ever refuted.
    """
    y_full = profile.detach().double().cpu().numpy() if isinstance(profile, torch.Tensor) else np.asarray(profile, float)
    n_full = y_full.shape[0]
    tv = None if truth is None else np.r_[_vec(truth), float(truth.reversed)]
    ledger: list[dict] = []
    H = _grid(config, config.levels[0].step)
    H = H[_in_box(H, config)]
    box_size = len(H)
    box_refuted = 0.0
    for k, lvl in enumerate(config.levels):
        t0 = time.time()
        cost, y, q = _level_costs(H, lvl, y_full, noise, config.order)
        i0 = int(np.argmin(cost))
        dof = max(1, 3 * (lvl.n_measure - (config.order + 1) - lvl.n_angle))
        birge = float(cost[i0] / dof)
        corr = max(1.0, noise.correlation * lvl.n_measure / n_full)
        delta = float(chi2.ppf(config.survival_quantile, 3) * max(1.0, birge) * corr)
        keep = cost <= cost[i0] + delta
        refuted_share = float(1.0 - keep.mean())
        if k == 0:
            box_refuted = refuted_share
        row = {
            "level": k + 1, "n_measure": lvl.n_measure, "n_angle": lvl.n_angle,
            "step": list(lvl.step), "tested": int(len(H)), "kept": int(keep.sum()),
            "refuted": int((~keep).sum()), "refuted_share": refuted_share,
            "best_chi2": float(cost[i0]), "birge": birge, "delta_chi2": delta,
            "seconds": round(time.time() - t0, 2),
            "best": _hyp(H[i0, :3], H[i0, 3]).as_dict(),
            "reversed_survivors": int(H[keep, 3].sum()),
            "forward_survivors": int((1 - H[keep, 3]).sum()),
        }
        if tv is not None:
            cov = np.all(np.abs(H[keep, :3] - tv[:3]) <= np.asarray(lvl.step) * 1.0001, axis=1) & (H[keep, 3] == tv[3])
            row["truth_still_covered"] = bool(cov.any())
        ledger.append(row)
        if log:
            log(f"level {k+1}: tested {row['tested']} kept {row['kept']} birge {birge:.2f}")
        alive, alive_cost = H[keep], cost[keep]
        last = (lvl, dof, birge, delta, y, q, corr)
        if k + 1 == len(config.levels):
            break
        if refuted_share < config.refine_if_refuted:
            row["stopped"] = ("too little refuted to refine: the geometry valley is wider than this "
                              "level's grid, so finer cells would only subdivide undecided territory")
            break
        H = _children(alive, config.levels[k + 1].step)
        H = H[_in_box(H, config)]
        if len(H) > config.max_children:
            row["stopped"] = f"refinement would test {len(H)} > {config.max_children} hypotheses"
            break
    lvl, dof, birge, delta, y, q, corr = last
    best_chi2 = float(alive_cost.min())
    bg = background_only_chi2(y, noise.sigma, q)
    # Nested test: the corner model adds 3 geometry parameters AND n_angle free
    # radiance bins per channel over background-only, so its threshold uses all of them.
    k_extra = 3 + 3 * lvl.n_angle
    bg_threshold = float(chi2.ppf(config.survival_quantile, k_extra) * max(1.0, birge) * corr)
    background_delta = float((bg - best_chi2) / bg_threshold)
    background_refuted = bool(background_delta > 1.0)
    reversed_refuted = bool(alive[:, 3].sum() == 0)
    adequate = bool(birge <= config.adequacy_birge)
    survivors = [_hyp(h[:3], h[3]) for h in alive]
    best = survivors[int(np.argmin(alive_cost))]
    decided = bool(reversed_refuted and box_refuted >= 0.9)
    if not adequate:
        status = "refuted-model"
        verdict = (f"No corner geometry in the search box explains this boundary: the best fit leaves "
                   f"{birge:.1f}x the photo's own noise. The corner model is refuted for this strip.")
    elif not background_refuted:
        status = "no-signal"
        verdict = ("Weak evidence: no occluder signature above this boundary's own noise: a constant plus a ramp explains it "
                   "as well as any corner geometry. The photo does not constrain the hidden side here.")
    elif decided:
        status = "decided-geometry"
        verdict = (f"Occluder signature present ({background_delta:.0f}x the survival threshold). "
                   f"{len(survivors)} geometries survive; {100*box_refuted:.0f}% of the search box and the "
                   f"reversed edge are refuted.")
    else:
        status = "undecided-geometry"
        verdict = (f"Occluder signature present ({background_delta:.0f}x the survival threshold), but the photo "
                   f"does not decide the corner geometry: {100*box_refuted:.0f}% of the search box is refuted, "
                   f"reversed edge {'refuted' if reversed_refuted else 'not refuted'}. Hidden angles are "
                   f"defined only up to an assumed geometry.")
    return FunnelResult(best, survivors, alive_cost.tolist(), ledger, best_chi2, dof, birge, bg,
                        background_delta, background_refuted, reversed_refuted, adequate,
                        box_refuted, decided, noise, verdict, status)


def funnel_from_image(image: torch.Tensor, region: BoundaryRegion, *, n_measure: int = 64,
                      smooth_sigma: float = 1.5, config: FunnelConfig = FunnelConfig(),
                      truth: CornerHypothesis | None = None) -> tuple[FunnelResult, torch.Tensor]:
    profile = extract_boundary_profile(image, region, n_measure=n_measure, smooth_sigma=smooth_sigma)
    noise = split_half_noise(image, region, n_measure=n_measure, smooth_sigma=smooth_sigma, order=config.order)
    return run_funnel(profile, noise, config, truth=truth), profile


# --------------------------------------------------- calibrated latent posterior --
_CONT = ("theta", "width", "height", "rgb", "brightness")


def _latent_matrix(z) -> np.ndarray:
    """[N, 7] continuous coordinates of a PhysicalLatent batch."""
    return np.concatenate([
        z.theta.double().numpy()[:, None], z.width.double().numpy()[:, None],
        z.height.double().numpy()[:, None], z.rgb.double().numpy(),
        z.brightness.double().numpy()[:, None]], 1)


def _box() -> tuple[np.ndarray, np.ndarray]:
    from .latent import PRIOR_RANGES as P
    lo = np.array([P["theta"][0], P["width"][0], P["height"][0]] + [P["rgb"][0]] * 3 + [P["brightness"][0]])
    hi = np.array([P["theta"][1], P["width"][1], P["height"][1]] + [P["rgb"][1]] * 3 + [P["brightness"][1]])
    return lo, hi


def _from_matrix(x: np.ndarray, shape: np.ndarray):
    from .latent import PhysicalLatent
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)
    return PhysicalLatent(t(x[:, 0]), t(x[:, 1]), t(x[:, 2]), t(x[:, 3:6]), t(x[:, 6]),
                          torch.as_tensor(shape, dtype=torch.long))


@dataclass
class CalibratedPosterior:
    weights: np.ndarray            # [n_pool] importance weights, marginal over geometries
    theta: np.ndarray              # [n_pool] candidate hidden angles
    pool_size: int
    ess: float
    temperature: float
    noise_correlation: float
    mean_theta: float
    std_theta: float
    interval90: tuple[float, float]
    interval68: tuple[float, float]
    mean_rgb: np.ndarray
    geometry_weights: np.ndarray   # [G] posterior mass of each geometry
    best_chi2: float

    def covers(self, theta: float, level: int = 90) -> bool:
        lo, hi = self.interval90 if level == 90 else self.interval68
        return bool(lo <= theta <= hi)

    def summary(self) -> dict:
        return {"pool_size": self.pool_size, "ess": self.ess, "temperature": self.temperature,
                "noise_correlation": self.noise_correlation, "mean_theta": self.mean_theta,
                "std_theta": self.std_theta, "interval90": list(self.interval90),
                "interval68": list(self.interval68), "mean_rgb": self.mean_rgb.tolist(),
                "best_chi2": self.best_chi2,
                "geometry_ess": float(1.0 / (self.geometry_weights ** 2).sum())}


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(x)
    cw = np.cumsum(w[o]); cw = cw / cw[-1]
    return float(np.interp(q, cw, x[o]))


def _chi2_table(y_white, noise, latent, scene, kernels_p, q, chunk=1024):
    """chi2[g, n]: shared non-negative exposure gain, background projected out."""
    from .scene import render_hidden
    m = y_white.shape[0]
    n_angle = kernels_p[0].shape[1]
    rt = CornerTransport(TransportConfig(n_measure=m, n_angle=n_angle))
    out = np.empty((len(kernels_p), latent.batch_size))
    for a in range(0, latent.batch_size, chunk):
        sub = latent.index(torch.arange(a, min(a + chunk, latent.batch_size)))
        R = rt._angular_radiance(render_hidden(sub, scene)).double().numpy()     # [n, J, 3]
        for gi, Kp in enumerate(kernels_p):
            pred = np.einsum("mj,njc->nmc", Kp, R) / noise.sigma[None, None, :]
            num = (pred * y_white[None]).sum((1, 2))
            den = (pred * pred).sum((1, 2))
            gain = np.clip(num / np.maximum(den, 1e-300), 0.0, None)
            out[gi, a:a + len(R)] = ((y_white[None] - gain[:, None, None] * pred) ** 2).sum((1, 2))
    return out


def calibrated_posterior(profile, noise: NoiseEstimate, scene, candidates,
                         geometries: list[CornerHypothesis], *, order: int = 1,
                         rounds: int = 5, per_round: int = 2000, centres: int = 64,
                         defensive: float = 0.2, seed: int = 0,
                         temperature: float | None = None) -> CalibratedPosterior:
    """Posterior over hidden scenes, marginalised over a set of corner geometries.

    * likelihood: Gaussian noise from the photo (split-half), constant + ramp
      removed per channel, one shared non-negative exposure gain;
    * correlated noise: chi-square divided by the noise correlation length;
    * temperature: Birge ratio of the best (candidate, geometry) pair, so a
      model that cannot reach the noise floor is tempered instead of certain;
    * sampling: population Monte Carlo. Round 0 is ``candidates`` (prior draws);
      later rounds draw from a defensive mixture of the prior and Gaussians around
      resampled particles. All draws are reweighted by the deterministic-mixture
      rule ``w = L p / (sum_r n_r q_r / N)``, so the weights stay exact.

    ``geometries`` should be the user's (or the generator's) geometry prior,
    after removing what the funnel refutes; the photo alone does not pick one.
    """
    y = profile.detach().double().cpu().numpy() if isinstance(profile, torch.Tensor) else np.asarray(profile, float)
    m = y.shape[0]
    q = background_basis(m, order)
    y_white = project_out(y, q) / noise.sigma[None, :]
    kernels_p = [project_out(g.kernel(m, 64), q) for g in geometries]
    lo, hi = _box()
    rng = np.random.default_rng(seed)
    vol = float(np.prod(hi - lo))
    log_prior = -math.log(vol) + math.log(0.5)              # uniform box x fair shape code

    X = _latent_matrix(candidates)
    S = candidates.shape_code.numpy().astype(int)
    chis = _chi2_table(y_white, noise, candidates, scene, kernels_p, q)
    proposals = [("prior", len(X), None)]
    dof = max(1, 3 * (m - (order + 1)) - 1)

    def weights(X, S, chis):
        T = max(1.0, float(chis.min()) / dof) if temperature is None else float(temperature)
        ll = -chis / (2.0 * T * noise.correlation)                     # [G, N]
        logL = np.logaddexp.reduce(ll, axis=0) - math.log(len(geometries))
        inside = np.all((X >= lo - 1e-12) & (X <= hi + 1e-12), 1)
        # deterministic-mixture proposal density
        logq_terms = []
        for kind, n, par in proposals:
            if kind == "prior":
                logq_terms.append(math.log(n) + np.where(inside, log_prior, -np.inf))
            else:
                C, Sc, sd, n_c = par
                d = (X[:, None, :] - C[None]) / sd
                lg = -0.5 * (d * d).sum(-1) - np.log(sd).sum() - 0.5 * X.shape[1] * math.log(2 * math.pi)
                ls = np.where(S[:, None] == Sc[None], math.log(0.9), math.log(0.1))
                comp = np.logaddexp.reduce(lg + ls, axis=1) - math.log(len(C))
                mix = np.logaddexp(math.log(defensive) + np.where(inside, log_prior, -np.inf),
                                   math.log(1 - defensive) + comp)
                logq_terms.append(math.log(n) + mix)
        logq = np.logaddexp.reduce(np.stack(logq_terms), axis=0) - math.log(sum(p[1] for p in proposals))
        logw = np.where(inside, logL + log_prior - logq, -np.inf)
        logw -= logw.max()
        w = np.exp(logw); w /= w.sum()
        return w, T, ll

    for r in range(rounds):
        w, T, _ = weights(X, S, chis)
        idx = rng.choice(len(X), size=centres, p=w)
        mu = (w[:, None] * X).sum(0)
        sd = np.sqrt((w[:, None] * (X - mu) ** 2).sum(0))
        sd = np.maximum(sd * 0.5, 0.01 * (hi - lo))
        C, Sc = X[idx], S[idx]
        n_mix = per_round
        k = rng.integers(0, centres, n_mix)
        use_prior = rng.random(n_mix) < defensive
        Xn = C[k] + rng.standard_normal((n_mix, X.shape[1])) * sd
        Sn = np.where(rng.random(n_mix) < 0.9, Sc[k], 1 - Sc[k])
        Xn[use_prior] = lo + (hi - lo) * rng.random((use_prior.sum(), X.shape[1]))
        Sn[use_prior] = rng.integers(0, 2, use_prior.sum())
        Xc = np.clip(Xn, lo, hi)
        keep_in = np.all(Xn == Xc, 1)                     # out-of-box draws keep zero weight
        new = _from_matrix(Xc, Sn)
        chis_new = _chi2_table(y_white, noise, new, scene, kernels_p, q)
        chis_new[:, ~keep_in] = np.inf
        Xn[~keep_in] = Xc[~keep_in] + 1e3                 # mark outside so prior density is zero
        proposals.append(("mix", n_mix, (C, Sc, sd, n_mix)))
        X = np.concatenate([X, Xn]); S = np.concatenate([S, Sn]); chis = np.concatenate([chis, chis_new], 1)

    w, T, ll = weights(X, S, chis)
    th = X[:, 0]
    finite = np.isfinite(chis)
    mt = float((w * th).sum())
    st = float(np.sqrt(max(0.0, (w * (th - mt) ** 2).sum())))
    ok = w > 0
    llk = ll[:, ok]
    post_g = np.exp(llk - llk.max(0, keepdims=True))
    post_g /= post_g.sum(0, keepdims=True)
    gw = (post_g * w[ok][None]).sum(1); gw = gw / gw.sum()
    return CalibratedPosterior(
        w, th, len(X), float(1.0 / (w ** 2).sum()), T, noise.correlation, mt, st,
        (weighted_quantile(th, w, 0.05), weighted_quantile(th, w, 0.95)),
        (weighted_quantile(th, w, 0.16), weighted_quantile(th, w, 0.84)),
        (w[:, None] * X[:, 3:6]).sum(0), gw, float(chis[finite].min()),
    )


def surviving_geometries(profile, noise: NoiseEstimate, geometries: list[CornerHypothesis],
                         funnel: FunnelResult, config: FunnelConfig = FunnelConfig()) -> list[CornerHypothesis]:
    """Drop geometries the photo refutes, at the funnel's last-level resolution and threshold."""
    row = funnel.ledger[-1]
    lvl = config.levels[row["level"] - 1]
    y_full = profile.detach().double().cpu().numpy() if isinstance(profile, torch.Tensor) else np.asarray(profile, float)
    y = _resample(y_full, lvl.n_measure)
    q = background_basis(lvl.n_measure, config.order)
    keep = [g for g in geometries
            if nn_misfit(g.funnel_kernel(lvl.n_measure, lvl.n_angle), y, noise.sigma, q)
            <= row["best_chi2"] + row["delta_chi2"]]
    return keep or list(geometries)


def hypothesis_survives(profile, noise: NoiseEstimate, hypothesis: CornerHypothesis,
                        funnel: FunnelResult, config: FunnelConfig = FunnelConfig()) -> bool:
    """Whether one geometry (e.g. the default) is still allowed by the photo."""
    row = funnel.ledger[-1]
    lvl = config.levels[row["level"] - 1]
    y_full = profile.detach().double().cpu().numpy() if isinstance(profile, torch.Tensor) else np.asarray(profile, float)
    y = _resample(y_full, lvl.n_measure)
    q = background_basis(lvl.n_measure, config.order)
    cost = nn_misfit(hypothesis.funnel_kernel(lvl.n_measure, lvl.n_angle), y, noise.sigma, q)
    return bool(cost <= row["best_chi2"] + row["delta_chi2"])
