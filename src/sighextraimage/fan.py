"""The 2-D penumbra fan: locate a wall corner from the light on the floor beside it.

Physics (the corner camera, Bouman et al. 2017; 2-D extension Seidel et al. 2020).
A vertical wall edge stands at the corner point O on the floor. A floor point at
angle phi around O (measured from the wall) receives light from the hidden scene
only at hidden angles theta < phi; the edge occludes the rest. So, apart from a
slow radial falloff, the extra light on the floor depends only on phi:

    I(x) = ambient(x) + sum_theta L(theta) * S((phi(x) - theta) / p) * g(r(x)).

Iso-intensity lines are rays from O. A camera maps the floor to the image by a
homography, which maps lines to lines. In the photo the rays therefore stay lines
through one point, the image of O. Perspective only reparametrises the angle
monotonically, and that is absorbed by free radiance bins. So the fan model
holds in pixel coordinates without camera calibration. Locating the apex is a
2-parameter search, just as Varjoluotain locates its plate.

A hypothesis is (apex x, apex y, which side is hidden, penumbra). It is refuted
when no non-negative angular radiance, with a two-term radial profile, explains
the photo's log-light beyond its own noise. The search runs as a funnel:
coarse blocks over a coarse grid, then finer blocks around unrefuted cells.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
import time

import numpy as np
from scipy.optimize import nnls
from scipy.stats import chi2


# ================================================================ rendering ===
def homography_from_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 H with dst ~ H src (DLT, 4+ point pairs)."""
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows, float))
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def smooth_field(shape, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance Gaussian random field with correlation length ~ scale pixels."""
    h, w = shape
    f = rng.standard_normal((h, w))
    ky = np.fft.fftfreq(h)[:, None]; kx = np.fft.fftfreq(w)[None, :]
    F = np.fft.fft2(f) * np.exp(-0.5 * (2 * np.pi * scale) ** 2 * (kx ** 2 + ky ** 2))
    out = np.real(np.fft.ifft2(F))
    return (out - out.mean()) / (out.std() + 1e-12)


def _linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 1e-8), 1 / 2.4) - 0.055)


def _srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


@dataclass
class FanScene:
    image_srgb: np.ndarray          # [H, W, 3] in [0,1], 8-bit quantised
    apex: np.ndarray                # image coordinates (x, y) of the corner O
    hidden_ccw: bool                # orientation flag in FanHypothesis's convention
    homography: np.ndarray          # floor (u, v) -> image (x, y)
    sources: list[dict]             # hidden sources: theta, width, distance, rgb
    leak_fraction: float            # peak hidden light / ambient on the floor
    source_image_angles: list[float] = field(default_factory=list)
    floor_mask: np.ndarray | None = None   # [H, W] pixels that see the visible floor


def render_fan_scene(seed: int, *, size: int = 256, leak: float = 0.03, n_sources: int | None = None,
                     noise_std: float = 0.002, texture: float = 0.03, null: bool = False,
                     quantize: bool = True) -> FanScene:
    """Synthetic photo of the floor beside a wall corner, hidden scene round the corner.

    Floor coordinates: corner O at (0, 0); the occluding wall occupies phi <= 0, so
    the visible floor is phi in (0, pi). A pinhole camera stands in the room
    (distance 2-3.5 from O, height 1-1.7, random roll) and looks down at a floor
    point near the corner, as in a corner-camera photo. Pixels that do not see the
    visible floor (wall, beyond the horizon) are masked out.
    """
    rng = np.random.default_rng(seed)
    psi, dist_c, hgt = rng.uniform(0.6, 1.4), rng.uniform(2.0, 3.5), rng.uniform(1.0, 1.7)
    cam = np.array([dist_c * math.cos(psi), dist_c * math.sin(psi), hgt])
    tr, ta = rng.uniform(0.45, 0.85), rng.uniform(0.35, 1.1)
    target = np.array([tr * math.cos(ta), tr * math.sin(ta), 0.0])
    zc = target - cam; zc /= np.linalg.norm(zc)
    xc = np.cross(zc, [0.0, 0.0, 1.0]); xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    roll = rng.uniform(0, 2 * math.pi)
    xr = math.cos(roll) * xc + math.sin(roll) * yc
    yr = -math.sin(roll) * xc + math.cos(roll) * yc
    R = np.stack([xr, yr, zc])
    f = (size / 2) * np.linalg.norm(target - cam) / rng.uniform(0.6, 0.9)
    K = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]])
    H = K @ R @ np.array([[1, 0, -cam[0]], [0, 1, -cam[1]], [0, 0, -cam[2]]])
    H = H / H[2, 2]
    Hi = np.linalg.inv(H)
    ys, xs = np.mgrid[0:size, 0:size]
    pix = np.c_[xs.ravel() + 0.5, ys.ravel() + 0.5]
    hom = np.c_[pix, np.ones(len(pix))] @ Hi.T
    fl = hom[:, :2] / hom[:, 2:3]
    # in front of the camera: the floor point must project back with positive depth
    depth = ((np.c_[fl, np.zeros(len(fl))] - cam) @ zc)
    r = np.hypot(fl[:, 0], fl[:, 1]); phi = np.arctan2(fl[:, 1], fl[:, 0])
    floor_mask = (depth > 0) & (phi > 0.0) & (r > 0.03) & (r < 6.0)
    # hidden sources
    n_src = int(rng.integers(1, 4)) if n_sources is None else n_sources
    sources = []
    leak_img = np.zeros((size * size, 3))
    for _ in range(n_src):
        th = float(rng.uniform(0.15, 1.35)); wd = float(rng.uniform(0.03, 0.15))
        dist = float(rng.uniform(1.5, 3.5)); rgb = rng.uniform(0.2, 1.0, 3)
        sources.append({"theta": th, "width": wd, "distance": dist, "rgb": rgb.tolist()})
        # extended source: integrate a smooth edge over its angular width
        vis = 0.5 * (1 + np.tanh((phi - th) / (0.5 * wd + 0.01)))
        fall = 1.0 / (1.0 + (r / dist) ** 2)
        leak_img += (vis * fall)[:, None] * rgb[None, :]
    # ambient light: constant + smooth gradient across the floor
    g = rng.uniform(-0.25, 0.25, 2)
    ambient = 0.35 * (1 + g[0] * (pix[:, 0] / size - 0.5) + g[1] * (pix[:, 1] / size - 0.5))
    if null:
        leak_img[:] = 0
    peak = leak_img.max() if leak_img.max() > 0 else 1.0
    leak_img = leak_img / peak * leak * 0.35
    albedo = np.array([0.62, 0.58, 0.52])[None, :] * np.exp(
        texture * smooth_field((size, size), rng.uniform(1.5, 4.0), rng).ravel())[:, None]
    lin = albedo * (ambient[:, None] + leak_img)
    lin[~floor_mask] = np.array([0.30, 0.30, 0.31]) * (0.8 + 0.1 * (pix[~floor_mask, 1:2] / size))
    lin = lin + noise_std * rng.standard_normal(lin.shape)
    srgb = _linear_to_srgb(lin)
    if quantize:
        srgb = np.round(srgb * 255.0) / 255.0
    apex = apply_h(H, np.zeros((1, 2)))[0]
    # orientation in the image: hidden side is where phi -> 0 (the wall direction)
    wall_pt = apply_h(H, np.array([[1.0, 0.0]]))[0]; open_pt = apply_h(H, np.array([[0.0, 1.0]]))[0]
    # the kernel cumulates light toward increasing image angle; when wall->open already
    # increases the angle, no flip is needed (flag False)
    hidden_ccw = not _ccw_from(apex, wall_pt, open_pt)
    src_angles = [float(_image_angle(apex, apply_h(H, np.array([[math.cos(s["theta"]), math.sin(s["theta"])]]))[0]))
                  for s in sources]
    return FanScene(srgb.reshape(size, size, 3), apex, hidden_ccw, H, sources,
                    0.0 if null else leak, src_angles, floor_mask.reshape(size, size))


def _image_angle(apex, pt) -> float:
    return math.atan2(pt[1] - apex[1], pt[0] - apex[0])


def _ccw_from(apex, wall_pt, open_pt) -> bool:
    """True when sweeping from the wall ray to the open ray is counter-clockwise in image angle."""
    a = _image_angle(apex, wall_pt); b = _image_angle(apex, open_pt)
    return ((b - a + math.pi) % (2 * math.pi) - math.pi) > 0


# ============================================================== measurement ===
@dataclass
class BlockMeasurement:
    y: np.ndarray        # [N, 3] mean linear light per block
    xy: np.ndarray       # [N, 2] block centres (image px)
    sigma: np.ndarray    # [3] per-block noise std (log units)
    correlation: float   # noise correlation length in blocks (>= 1)
    block: int
    shape: tuple[int, int]
    index: np.ndarray    # flat indices of the used blocks in the [nh, nw] grid


def measure_blocks(image_srgb: np.ndarray, block: int, mask: np.ndarray | None = None) -> BlockMeasurement:
    """Mean linear light in block x block cells; noise from a 5-point Laplacian.

    The Laplacian of the block image cancels anything locally linear (ambient
    ramps, slow fan structure), so its MAD estimates the block-to-block noise,
    texture included. That is deliberately conservative: albedo texture at the
    block scale is a nuisance the fan cannot explain, and it is counted as noise.
    """
    lin = _srgb_to_linear(np.asarray(image_srgb, float))
    h, w = lin.shape[:2]
    nh, nw = h // block, w // block
    # Linear light: hidden light ADDS to room light, so the fan is linear in it.
    # (Log light would divide the leak by the spatially varying ambient.)
    L = lin[:nh * block, :nw * block].reshape(nh, block, nw, block, 3).mean((1, 3))
    ok = np.ones((nh, nw), bool)
    if mask is not None:
        ok = mask[:nh * block, :nw * block].reshape(nh, block, nw, block).all((1, 3))
    lap = (4 * L[1:-1, 1:-1] - L[:-2, 1:-1] - L[2:, 1:-1] - L[1:-1, :-2] - L[1:-1, 2:])
    okl = ok[1:-1, 1:-1] & ok[:-2, 1:-1] & ok[2:, 1:-1] & ok[1:-1, :-2] & ok[1:-1, 2:]
    d = lap[okl] if okl.any() else lap.reshape(-1, 3)
    sigma = 1.4826 * np.median(np.abs(d - np.median(d, 0)), 0) / math.sqrt(20.0)
    sigma = np.maximum(sigma, (1.0 / 255.0) / math.sqrt(12.0) / block * 0.1)  # 8-bit floor, linear units
    # correlation from lag-1 autocorrelation of the Laplacian-whitened residual
    lag = (lap[:, 1:] * lap[:, :-1]).sum() / max((lap * lap).sum(), 1e-30)
    corr = float(max(1.0, 1.0 + 2.0 * max(0.0, (lag + 0.25) / 0.75)))  # Laplacian of white noise has lag-1 = -0.25
    ys, xs = np.mgrid[0:nh, 0:nw]
    xy = np.c_[(xs.ravel() + 0.5) * block, (ys.ravel() + 0.5) * block]
    return BlockMeasurement(L.reshape(-1, 3)[ok.ravel()], xy[ok.ravel()], sigma, corr, block, (nh, nw),
                            np.flatnonzero(ok.ravel()))


def background_basis_2d(xy: np.ndarray, size: float, order: int = 1) -> np.ndarray:
    u = xy[:, 0] / size * 2 - 1; v = xy[:, 1] / size * 2 - 1
    cols = [np.ones(len(xy)), u, v]
    if order >= 2:
        cols += [u * u, v * v, u * v]
    q, _ = np.linalg.qr(np.stack(cols, 1))
    return q


def _proj(m, q):
    return m - q @ (q.T @ m)


# =================================================================== model ====
@dataclass(frozen=True)
class FanHypothesis:
    apex_x: float
    apex_y: float
    hidden_ccw: bool         # hidden-scene side lies counter-clockwise of the open side
    penumbra: float          # edge softness, radians of image angle

    def as_dict(self):
        return asdict(self)


def fan_kernel(h: FanHypothesis, xy: np.ndarray, n_bins: int, size: float, block: float = 1.0,
               sub: int = 4, n_radial: int = 2, amplitude: str = "radial") -> np.ndarray:
    """[N, n_radial*n_bins] columns: angular bin j x non-negative radial hat in log radius.

    Each column is averaged over sub x sub points inside the block, because the
    measurement is a block average and blocks near the apex span many angles.
    Angles are unwrapped around the patch's mean direction; bins tile the span.
    """
    off = (np.arange(sub) + 0.5) / sub - 0.5
    ox, oy = np.meshgrid(off * block, off * block)
    P = xy[:, None, :] + np.stack([ox.ravel(), oy.ravel()], 1)[None]       # [N, S, 2]
    dx = P[..., 0] - h.apex_x; dy = P[..., 1] - h.apex_y
    c = math.atan2(dy.mean(), dx.mean())
    ang = (np.arctan2(dy, dx) - c + math.pi) % (2 * math.pi) - math.pi
    if h.hidden_ccw:
        ang = -ang                       # cumulate from the other side
    lo, hi = ang.min(), ang.max()
    pad = 3 * h.penumbra
    th = np.linspace(lo - pad, hi + pad, n_bins)
    r = np.log(np.maximum(np.hypot(dx, dy), 1.0))
    rho = (r - r.min()) / max(r.max() - r.min(), 1e-9)                       # log-radius in [0, 1]
    vis = 0.5 * (1 + np.tanh((ang[..., None] - th) / h.penumbra))            # [N, S, B]
    if amplitude == "plane":
        # Each edge's amplitude is a positive plane over the image: non-negative
        # combinations of (1+u), (1-u), (1+v), (1-v), u and v in [-1, 1].
        # Perspective makes the falloff behind an edge vary along it; the plane
        # absorbs that without letting radius-about-the-apex pin the position.
        u = (xy[:, 0] - xy[:, 0].min()) / max(np.ptp(xy[:, 0]), 1e-9) * 2 - 1
        v = (xy[:, 1] - xy[:, 1].min()) / max(np.ptp(xy[:, 1]), 1e-9) * 2 - 1
        vm = vis.mean(1)                                                      # [N, B]
        return np.concatenate([vm * (1 + u)[:, None], vm * (1 - u)[:, None],
                               vm * (1 + v)[:, None], vm * (1 - v)[:, None]], 1)
    nodes = np.linspace(0.0, 1.0, n_radial)
    width = 1.0 / max(n_radial - 1, 1)
    cols = []
    for nd in nodes:                                                          # hat functions in log radius
        hat = np.clip(1.0 - np.abs(rho - nd) / width, 0.0, 1.0)
        cols.append((vis * hat[..., None]).mean(1))
    return np.concatenate(cols, 1)


def fan_misfit(h: FanHypothesis, m: BlockMeasurement, q: np.ndarray, n_bins: int, size: float,
               return_fit: bool = False, n_radial: int = 2, amplitude: str = "radial"):
    K = _proj(fan_kernel(h, m.xy, n_bins, size, m.block, n_radial=n_radial, amplitude=amplitude), q)
    yp = _proj(m.y, q)
    total, coefs, fit = 0.0, [], np.zeros_like(yp)
    for c in range(3):
        x, rn = nnls(K, yp[:, c], maxiter=30 * K.shape[1])
        total += (rn / m.sigma[c]) ** 2
        coefs.append(x); fit[:, c] = K @ x
    if return_fit:
        return total, np.stack(coefs, 1), fit, yp
    return total


# ================================================================== funnel ====
@dataclass(frozen=True)
class FanLevel:
    block: int
    n_bins: int
    step: float              # apex grid step, px
    penumbras: tuple[float, ...] = (0.03,)
    cap: int = 400           # most hypotheses carried to the next level (lowest chi2 first)


@dataclass(frozen=True)
class FanConfig:
    box: float = 1.0                          # apex search box: image +- box*size around the image
    levels: tuple[FanLevel, ...] = (
        FanLevel(16, 24, 16.0, (0.03,), 600),
        FanLevel(8, 48, 4.0, (0.03,), 200),
        FanLevel(8, 48, 2.0, (0.01, 0.03, 0.08), 400),
    )
    survival_quantile: float = 0.99
    adequacy_birge: float = 4.0
    max_children: int = 6000
    bg_order: int = 2
    model_error: float = 0.20     # systematic floor: fraction of the projected signal rms per channel
    located_px: float = 16.0      # survivor cloud's long axis below this: corner located (2 blocks)
    detect_threshold: float = 3.0 # background refuted when its excess chi2 exceeds this many nested thresholds


@dataclass
class FanResult:
    best: FanHypothesis
    survivors: list[FanHypothesis]
    ledger: list[dict]
    birge: float
    background_delta: float
    background_refuted: bool
    orientation_decided: bool
    apex_spread: float                        # px: max distance of a survivor from the best apex
    long_axis: float                          # px: survivor cloud along its main direction
    short_axis: float
    line_point: np.ndarray                    # survivor-cloud centre and main direction (image px)
    line_dir: np.ndarray
    status: str
    verdict: str
    panorama: np.ndarray | None = None        # [n_bins, 3] near-radiance at the best hypothesis
    panorama_angles: np.ndarray | None = None
    residue: np.ndarray | None = None         # [nh, nw] |residual|/sigma at the best hypothesis

    @property
    def temperature(self):
        return max(1.0, self.birge)

    def summary(self) -> dict:
        xs = np.array([[s.apex_x, s.apex_y] for s in self.survivors])
        return {"status": self.status, "verdict": self.verdict, "best": self.best.as_dict(),
                "n_survivors": len(self.survivors), "apex_spread_px": self.apex_spread,
                "long_axis_px": self.long_axis, "short_axis_px": self.short_axis,
                "line_point": self.line_point.tolist(), "line_dir": self.line_dir.tolist(),
                "survivor_box": [xs.min(0).tolist(), xs.max(0).tolist()],
                "orientation_decided": self.orientation_decided, "birge": self.birge,
                "background_delta": self.background_delta, "background_refuted": self.background_refuted,
                "ledger": self.ledger}


def _initial_grid(size: float, cfg: FanConfig, step: float, pens) -> list[FanHypothesis]:
    lo, hi = -cfg.box * size, (1 + cfg.box) * size
    ax = np.arange(lo, hi + 1e-9, step)
    out = []
    for x in ax:
        for y in ax:
            for o in (False, True):
                for p in pens:
                    out.append(FanHypothesis(float(x), float(y), o, p))
    return out


def _children(parents: list[FanHypothesis], step: float, size: float, pens) -> list[FanHypothesis]:
    seen, out = set(), []
    for h in parents:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                x, y = h.apex_x + dx * step, h.apex_y + dy * step
                for p in pens:
                    key = (round(x / step * 4), round(y / step * 4), h.hidden_ccw, p)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(FanHypothesis(float(x), float(y), h.hidden_ccw, p))
    return out


def locate_apex(image_srgb: np.ndarray, config: FanConfig = FanConfig(), *, mask: np.ndarray | None = None,
                truth: FanScene | None = None, log=None) -> FanResult:
    size = float(min(image_srgb.shape[:2]))
    ledger, alive = [], None
    for k, lvl in enumerate(config.levels):
        t0 = time.time()
        m = measure_blocks(image_srgb, lvl.block, mask)
        q = background_basis_2d(m.xy, size, config.bg_order)
        # Model-resolution floor: finite bins and a two-term radial profile leave a
        # residue that grows with the signal, not with the noise. Without it, a
        # bright fan turns tiny approximation errors into confident refutations.
        sig_rms = np.sqrt((_proj(m.y, q) ** 2).mean(0))
        m.sigma = np.sqrt(m.sigma ** 2 + (config.model_error * sig_rms) ** 2)
        H = (_initial_grid(size, config, lvl.step, lvl.penumbras) if alive is None
             else _children(alive, lvl.step, size, lvl.penumbras))
        if alive is not None and len(H) > config.max_children:
            ledger[-1]["stopped"] = f"refinement would test {len(H)} > {config.max_children} hypotheses"
            break
        cost = np.array([fan_misfit(h, m, q, lvl.n_bins, size) for h in H])
        i0 = int(np.argmin(cost))
        dof = max(1, 3 * (len(m.y) - q.shape[1] - 2 * lvl.n_bins))
        birge = float(cost[i0] / dof)
        delta = float(chi2.ppf(config.survival_quantile, 3) * max(1.0, birge) * m.correlation)
        keep = cost <= cost[i0] + delta
        n_unrefuted = int(keep.sum())
        if n_unrefuted > lvl.cap:                 # compute cap: carry the best-fitting unrefuted cells
            keep = np.zeros_like(keep); keep[np.argsort(cost)[: lvl.cap]] = True
        alive = [h for h, kk in zip(H, keep) if kk]
        row = {"level": k + 1, "block": lvl.block, "n_bins": lvl.n_bins, "step_px": lvl.step,
               "n_measure": int(len(m.y)), "tested": len(H), "kept": int(keep.sum()),
               "unrefuted": n_unrefuted, "capped": bool(n_unrefuted > lvl.cap),
               "refuted": int(len(H) - n_unrefuted), "best_chi2": float(cost[i0]), "birge": birge,
               "delta_chi2": delta, "noise_sigma": m.sigma.tolist(), "noise_correlation": m.correlation,
               "best": H[i0].as_dict(), "seconds": round(time.time() - t0, 2),
               "orientations_alive": sorted({h.hidden_ccw for h in alive})}
        if truth is not None:
            # covered: some surviving cell (same orientation) contains the true apex
            row["truth_still_covered"] = any(
                abs(h.apex_x - truth.apex[0]) <= lvl.step * 1.0001 and abs(h.apex_y - truth.apex[1]) <= lvl.step * 1.0001
                and h.hidden_ccw == truth.hidden_ccw for h in alive)
            row["truth_nearest_survivor_px"] = float(min(
                math.hypot(h.apex_x - truth.apex[0], h.apex_y - truth.apex[1]) for h in alive))
            row["best_apex_error_px"] = float(math.hypot(H[i0].apex_x - truth.apex[0], H[i0].apex_y - truth.apex[1]))
        ledger.append(row)
        if log:
            log(f"level {k+1}: tested {len(H)} kept {int(keep.sum())} birge {birge:.2f} best {H[i0]}")
        last = (lvl, m, q, cost[keep], delta, birge, dof)
    lvl, m, q, alive_cost, delta, birge, dof = last
    best = alive[int(np.argmin(alive_cost))]
    total, coefs, fit, yp = fan_misfit(best, m, q, lvl.n_bins, size, return_fit=True)
    bg = float(((yp / m.sigma[None, :]) ** 2).sum())
    k_extra = 4 + 3 * 2 * lvl.n_bins
    bg_thr = float(chi2.ppf(config.survival_quantile, k_extra) * max(1.0, birge) * m.correlation)
    bg_delta = (bg - total) / bg_thr
    bg_ref = bool(bg_delta > config.detect_threshold)
    orient = len({h.hidden_ccw for h in alive}) == 1
    spread = float(max(math.hypot(h.apex_x - best.apex_x, h.apex_y - best.apex_y) for h in alive))
    pts = np.array([[h.apex_x, h.apex_y] for h in alive])
    if len(pts) > 2:
        ev, evec = np.linalg.eigh(np.cov(pts.T))
        axes = (float(4 * math.sqrt(max(ev[1], 0))), float(4 * math.sqrt(max(ev[0], 0))))  # ~ +-2 sd extents
        line_dir = evec[:, 1]
    else:
        axes, line_dir = (0.0, 0.0), np.array([1.0, 0.0])
    long_axis = max(axes[0], float(np.ptp(pts @ line_dir)) if len(pts) > 1 else 0.0)
    short_axis = axes[1]
    line_point = pts.mean(0)
    adequate = birge <= config.adequacy_birge
    resid = np.full(m.shape[0] * m.shape[1], np.nan)
    resid[m.index] = np.sqrt((((yp - fit) / m.sigma[None, :]) ** 2).mean(1))
    if not adequate:
        status, verdict = "refuted-model", (f"No corner fan explains this floor: the best fit leaves {birge:.1f}x "
                                            f"the image's own noise.")
    elif not bg_ref:
        status, verdict = "no-signal", ("Weak evidence: no penumbra fan above this image's noise; light gradients "
                                        "alone explain it. The corner cannot be located from this photo.")
    elif long_axis <= config.located_px:
        status = "located" if orient else "located-side-undecided"
        verdict = (f"Penumbra fan present ({bg_delta:.1f}x threshold). Corner located at "
                   f"({best.apex_x:.0f}, {best.apex_y:.0f}) px; {len(alive)} hypotheses survive within "
                   f"{spread:.0f} px. Hidden side {'decided' if orient else 'NOT decided'}.")
    else:
        status = "on-a-line" if short_axis <= config.located_px else "undecided"
        verdict = (f"Penumbra fan present ({bg_delta:.1f}x threshold), but the corner is not pinned: surviving "
                   f"apexes spread {long_axis:.0f} px long and {short_axis:.0f} px wide. "
                   + ("They lie along one line: a single shadow edge fixes a line through the corner, "
                      "and a second edge is needed to triangulate it." if short_axis <= config.located_px else ""))
    return FanResult(best, alive, ledger, birge, float(bg_delta), bg_ref, orient, spread, long_axis, short_axis,
                     line_point, line_dir,
                     status, verdict,
                     panorama=coefs[: lvl.n_bins] + coefs[lvl.n_bins: 2 * lvl.n_bins], panorama_angles=None,
                     residue=resid.reshape(m.shape))


def distance_to_line(result: FanResult, point) -> float:
    """Perpendicular distance (px) from a point to the survivor cloud's main line."""
    d = np.asarray(point, float) - result.line_point
    n = np.array([-result.line_dir[1], result.line_dir[0]])
    return float(abs(d @ n))


def load_photo(path: str, max_side: int = 256, region=None):
    """Photo -> (sRGB float image, floor mask). ``region`` = (x0, y0, x1, y1) image fractions of the floor."""
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if region is not None:
            x0, y0, x1, y1 = region
            im = im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
        sc = max_side / max(im.size)
        if sc < 1:
            im = im.resize((max(1, round(im.size[0] * sc)), max(1, round(im.size[1] * sc))), Image.Resampling.LANCZOS)
        arr = np.asarray(im, dtype=np.float64) / 255.0
    side = min(arr.shape[:2])
    arr = arr[:side, :side]
    return arr, np.ones(arr.shape[:2], bool)


def report_image(image_srgb: np.ndarray, result: FanResult, scale: int = 2, truth_apex=None, certificate=None):
    """Annotated report: photo with surviving corners (or line), residue map, hidden-angle panorama."""
    from PIL import Image, ImageDraw
    hgt, wid = image_srgb.shape[:2]
    if certificate is not None:
        # draw what the certificate decided instead of the funnel's fragment
        from types import SimpleNamespace
        result = SimpleNamespace(
            survivors=[SimpleNamespace(apex_x=x, apex_y=y) for x, y, _ in certificate.survivors],
            status=certificate.status, best=certificate.best, line_point=certificate.line_point,
            line_dir=certificate.line_dir, residue=result.residue, panorama=result.panorama)
    # pad just enough to show the surviving apexes (capped at one image size)
    pts = np.array([[h.apex_x, h.apex_y] for h in result.survivors] + [[0, 0], [wid, hgt]])
    need = max(0.0, -pts.min(), (pts[:, 0] - wid).max(), (pts[:, 1] - hgt).max())
    pad = int(min(wid, need + 24))
    canvas = Image.new("RGB", ((wid + 2 * pad) * scale, (hgt + 2 * pad) * scale + 70 * scale), (24, 24, 28))
    photo = Image.fromarray((np.clip(image_srgb, 0, 1) * 255).astype(np.uint8)).resize((wid * scale, hgt * scale))
    canvas.paste(photo, (pad * scale, pad * scale))
    d = ImageDraw.Draw(canvas)
    to = lambda x, y: ((x + pad) * scale, (y + pad) * scale)
    for hpt in result.survivors:
        cx, cy = to(hpt.apex_x, hpt.apex_y)
        d.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(255, 200, 60))
    if result.status in ("on-a-line", "undecided"):
        p0 = result.line_point - result.line_dir * 3 * wid; p1 = result.line_point + result.line_dir * 3 * wid
        d.line([to(*p0), to(*p1)], fill=(255, 120, 60), width=scale)
    if truth_apex is not None:
        tx, ty = to(*truth_apex)
        d.line([tx - 10, ty, tx + 10, ty], fill=(80, 255, 120), width=2)
        d.line([tx, ty - 10, tx, ty + 10], fill=(80, 255, 120), width=2)
    bx, by = to(result.best.apex_x, result.best.apex_y)
    d.ellipse([bx - 6, by - 6, bx + 6, by + 6], outline=(255, 60, 60), width=2)
    if result.residue is not None:
        res = np.nan_to_num(result.residue, nan=0.0)
        res = np.clip(res / 3.0, 0, 1)
        im = Image.fromarray((res * 255).astype(np.uint8)).resize((wid // 3 * scale, hgt // 3 * scale), Image.Resampling.NEAREST)
        canvas.paste(im.convert("RGB"), (4, 4))
    if result.panorama is not None:
        pano = np.clip(result.panorama / max(result.panorama.max(), 1e-12), 0, 1)
        strip = (np.repeat(pano[None], 16, 0) * 255).astype(np.uint8)
        strip_im = Image.fromarray(strip).resize(((wid + 2 * pad) * scale, 40 * scale), Image.Resampling.NEAREST)
        canvas.paste(strip_im, (0, (hgt + 2 * pad) * scale + 20 * scale))
    d.text((8 * scale, (hgt + 2 * pad) * scale + 4), result.status + ": hidden-angle panorama (fan bins)", fill=(230, 230, 230))
    return canvas


# ============================================================ certification ===
def bin_image_angles(h: FanHypothesis, xy: np.ndarray, n_bins: int, block: float = 1.0, sub: int = 4) -> np.ndarray:
    """Image angle (radians, atan2 convention) of each angular bin's edge, as built by fan_kernel."""
    off = (np.arange(sub) + 0.5) / sub - 0.5
    ox, oy = np.meshgrid(off * block, off * block)
    P = xy[:, None, :] + np.stack([ox.ravel(), oy.ravel()], 1)[None]
    dx = P[..., 0] - h.apex_x; dy = P[..., 1] - h.apex_y
    c = math.atan2(dy.mean(), dx.mean())
    ang = (np.arctan2(dy, dx) - c + math.pi) % (2 * math.pi) - math.pi
    if h.hidden_ccw:
        ang = -ang
    pad = 3 * h.penumbra
    th = np.linspace(ang.min() - pad, ang.max() + pad, n_bins)
    return c - th if h.hidden_ccw else c + th


def edge_clusters(contrib: np.ndarray, angles: np.ndarray, *, rel: float = 0.15, gap: float = 0.15) -> list[dict]:
    """Distinct edges in a fitted fan.

    ``contrib[j]`` is the energy bin j adds to the fitted image (after background
    removal), not its raw coefficient: a bin whose column the background nearly
    cancels can carry a huge coefficient and contribute nothing. Bins with
    >= ``rel`` of the strongest bin's energy are grouped into clusters separated
    by more than ``gap`` radians of image angle.
    """
    if contrib.max() <= 0:
        return []
    sig = np.flatnonzero(contrib >= rel * contrib.max())
    order = sig[np.argsort(angles[sig])]
    clusters, cur = [], [order[0]]
    for a, b in zip(order, order[1:]):
        if abs(angles[b] - angles[a]) > gap:
            clusters.append(cur); cur = [b]
        else:
            cur.append(b)
    clusters.append(cur)
    out = []
    for cl in clusters:
        w = contrib[cl]
        out.append({"angle": float((angles[cl] * w).sum() / w.sum()), "energy": float(w.sum()),
                    "bins": [int(i) for i in cl]})
    return sorted(out, key=lambda e: -e["energy"])


def bin_contributions(h: FanHypothesis, m: "BlockMeasurement", q: np.ndarray, n_bins: int, size: float,
                      coefs: np.ndarray, n_radial: int = 2) -> np.ndarray:
    Kp = _proj(fan_kernel(h, m.xy, n_bins, size, m.block, n_radial=n_radial), q)
    out = np.zeros(n_bins)
    for j in range(n_bins):
        cols = [j + k * n_bins for k in range(n_radial)]
        part = Kp[:, cols] @ coefs[cols]                 # [N, 3]
        out[j] = float(((part / m.sigma[None, :]) ** 2).sum())
    return out


@dataclass(frozen=True)
class CertifyConfig:
    block: int = 8
    n_bins: int = 48
    penumbras: tuple[float, ...] = (0.01, 0.03, 0.08)
    ring_dirs: int = 24                         # coarse ring: catches broad ambiguity
    ring_radii: tuple[float, ...] = (16.0, 32.0, 64.0, 128.0)
    valley_radius: float = 24.0                 # fine ring for the narrow single-edge valley
    valley_dirs: int = 72
    valley_refine_deg: float = 0.5
    walk_step: float = 12.0
    walk_max: float = 512.0
    walk_lateral: tuple[float, ...] = (-4.0, -2.0, 0.0, 2.0, 4.0)
    located_px: float = 16.0
    line_elongation: float = 4.0                # a survivor set this elongated is a line ...
    line_width_px: float = 32.0                 # ... if it is at most this wide
    edge_rel: float = 0.15
    edge_gap: float = 0.15


@dataclass
class Certificate:
    status: str
    verdict: str
    best: FanHypothesis
    survivors: list[tuple[float, float, bool]]   # (x, y, hidden_ccw) of every evaluated point within threshold
    evaluated: int
    threshold: float
    best_chi2: float
    long_axis: float
    short_axis: float
    edges: list[dict]
    line_point: np.ndarray
    line_dir: np.ndarray
    side_decided: bool
    valley_rise: float                          # chi2 rise along the best valley direction at valley_radius

    def summary(self) -> dict:
        return {"status": self.status, "verdict": self.verdict, "best": self.best.as_dict(),
                "n_survivors": len(self.survivors), "evaluated": self.evaluated,
                "threshold": self.threshold, "best_chi2": self.best_chi2,
                "long_axis_px": self.long_axis, "short_axis_px": self.short_axis,
                "edges": self.edges, "n_edges": len(self.edges),
                "line_point": self.line_point.tolist(), "line_dir": self.line_dir.tolist(),
                "side_decided": self.side_decided, "valley_rise": self.valley_rise}


def certify(image_srgb: np.ndarray, result: FanResult, config: CertifyConfig = CertifyConfig(), *,
            mask: np.ndarray | None = None, funnel_config: FanConfig = FanConfig()) -> Certificate:
    """Pay to check before claiming (Luotain's rule).

    The funnel returns a best apex and a cloud of survivors. That cloud can be
    a fragment: one shadow edge leaves a valley only 1-2 px wide along the
    edge's line, which a coarse grid steps over. Certification therefore:

    1. re-evaluates the survivors, the best apex on both hidden sides, and a
       coarse ring of probes with the finest model;
    2. finds the valley direction: chi2 on a fine ring (72 directions, then
       0.5 deg refinement) at ``valley_radius``, taking the cheaper of each
       antipodal pair because a single edge leaves a half-line;
    3. walks the valley both ways, re-centring sideways at every step, until
       the photo refutes it.

    The corner is reported as a point only if every surviving position lies
    within ``located_px``. Otherwise the line through the best apex along the
    valley is reported: the corner is somewhere on it.
    """
    size = float(min(image_srgb.shape[:2]))
    m = measure_blocks(image_srgb, config.block, mask)
    q = background_basis_2d(m.xy, size, funnel_config.bg_order)
    sig_rms = np.sqrt((_proj(m.y, q) ** 2).mean(0))
    m.sigma = np.sqrt(m.sigma ** 2 + (funnel_config.model_error * sig_rms) ** 2)
    cache: dict = {}

    def cost(x, y, side, pens=config.penumbras):
        key = (round(float(x), 2), round(float(y), 2), bool(side))
        if key not in cache or (pens is config.penumbras and cache[key][2] is not config.penumbras):
            vals = [(fan_misfit(FanHypothesis(float(x), float(y), bool(side), p), m, q, config.n_bins, size), p)
                    for p in pens]
            c, p = min(vals)
            if key in cache and cache[key][0] < c:
                return cache[key][0]
            cache[key] = (c, p, pens)
        return cache[key][0]

    b = result.best
    pts = {(h.apex_x, h.apex_y, h.hidden_ccw) for h in result.survivors}
    for side in (False, True):
        pts.add((b.apex_x, b.apex_y, side))
        for k in range(config.ring_dirs):
            a = 2 * math.pi * k / config.ring_dirs
            for r in config.ring_radii:
                pts.add((b.apex_x + r * math.cos(a), b.apex_y + r * math.sin(a), side))
    for p in pts:
        cost(*p)
    dof = max(1, 3 * (len(m.y) - q.shape[1] - 2 * config.n_bins))

    def threshold():
        best_c = min(v[0] for v in cache.values())
        birge = best_c / dof
        return best_c, best_c + float(chi2.ppf(funnel_config.survival_quantile, 3) * max(1.0, birge) * m.correlation)

    best_key = min(cache, key=lambda kk: cache[kk][0])
    bx, by, bside = best_key
    pen = (cache[best_key][1],)
    # fine valley direction (half-line aware)
    R = config.valley_radius

    def dir_cost(a):
        return min(cost(bx + R * math.cos(a), by + R * math.sin(a), bside, pen),
                   cost(bx - R * math.cos(a), by - R * math.sin(a), bside, pen))
    grid = np.linspace(0, math.pi, config.valley_dirs, endpoint=False)
    a0 = grid[int(np.argmin([dir_cost(a) for a in grid]))]
    span = math.pi / config.valley_dirs
    fine = np.arange(a0 - span, a0 + span + 1e-9, math.radians(config.valley_refine_deg))
    a_star = fine[int(np.argmin([dir_cost(a) for a in fine]))]
    valley_rise = dir_cost(a_star) - cache[best_key][0]
    # walk the valley both ways with lateral re-centring
    for sgn in (1.0, -1.0):
        d = np.array([math.cos(a_star), math.sin(a_star)]) * sgn
        nrm = np.array([-d[1], d[0]])
        cur = np.array([bx, by], float)
        t = 0.0
        while t < config.walk_max:
            t += config.walk_step
            nxt = cur + d * config.walk_step
            cand = [(cost(*(nxt + l * nrm), bside, pen), l) for l in config.walk_lateral]
            c, l = min(cand)
            _, thr = threshold()
            if c > thr:
                break
            cur = nxt + l * nrm
            if abs(l) > 0:                       # keep the walk aimed along the valley's measured course
                d = (cur - np.array([bx, by])) / np.linalg.norm(cur - np.array([bx, by])) * 1.0
                nrm = np.array([-d[1], d[0]])
    best_c, thr = threshold()
    best_key = min(cache, key=lambda kk: cache[kk][0])
    bh = FanHypothesis(best_key[0], best_key[1], best_key[2], cache[best_key][1])
    surv = [kk for kk, v in cache.items() if v[0] <= thr]
    P = np.array([[x, y] for x, y, _ in surv])
    if len(P) > 2:
        ev, evec = np.linalg.eigh(np.cov(P.T))
        long_axis = float(np.ptp(P @ evec[:, 1])); short_axis = float(np.ptp(P @ evec[:, 0]))
        pca_dir = evec[:, 1]
    else:
        long_axis = short_axis = 0.0; pca_dir = np.array([math.cos(a_star), math.sin(a_star)])
    _, coefs, _, _ = fan_misfit(bh, m, q, config.n_bins, size, return_fit=True)
    angles = bin_image_angles(bh, m.xy, config.n_bins, m.block)
    edges = edge_clusters(bin_contributions(bh, m, q, config.n_bins, size, coefs), angles,
                          rel=config.edge_rel, gap=config.edge_gap)
    side_decided = len({s for _, _, s in surv}) == 1
    lpoint = np.array([bh.apex_x, bh.apex_y])
    # line direction: the long survivor cloud when the walk mapped one, else the fine valley direction
    ldir = pca_dir if long_axis > 4 * config.located_px else np.array([math.cos(a_star), math.sin(a_star)])
    if long_axis <= config.located_px:
        status = "located" if side_decided else "located-side-undecided"
        verdict = (f"Corner located at ({bh.apex_x:.0f}, {bh.apex_y:.0f}) px: all {len(surv)} surviving positions "
                   f"lie within {max(long_axis, 1):.0f} px after {len(cache)} positions were checked, including "
                   f"the cheapest valley direction (rise {valley_rise:.0f} at {R:.0f} px).")
    elif short_axis <= config.located_px or (long_axis >= config.line_elongation * short_axis
                                              and short_axis <= config.line_width_px):
        status = "on-a-line"
        verdict = (f"The corner lies on a line through ({bh.apex_x:.0f}, {bh.apex_y:.0f}) px at image angle "
                   f"{math.degrees(math.atan2(ldir[1], ldir[0])):.0f} deg; surviving positions span "
                   f"{long_axis:.0f} px along it. One shadow edge fixes a line, not a point.")
    else:
        status = "undecided"
        verdict = (f"The fan is present but the corner is not pinned: {len(surv)} surviving positions span "
                   f"{long_axis:.0f} x {short_axis:.0f} px.")
    return Certificate(status, verdict, bh, surv, len(cache), float(thr), float(best_c), long_axis, short_axis,
                       edges, lpoint, ldir, side_decided, float(valley_rise))


def certified_distance_to_line(cert: Certificate, point) -> float:
    d = np.asarray(point, float) - cert.line_point
    n = np.array([-cert.line_dir[1], cert.line_dir[0]])
    return float(abs(d @ n))
