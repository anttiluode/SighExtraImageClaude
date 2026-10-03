# Gate 3 — locate the wall corner from the 2-D penumbra fan

Date: 2026-10-03
Status: preregistered before any held-out result was computed. Development used seeds 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 only. One check printed the held-out scenes' composition (7 with two or more distinct edges, 9 with one), not any result.

## Why

Gate 2 showed that a 1-D boundary strip cannot decide the corner geometry: remapping the strip's angle axis is absorbed by the hidden radiance. Varjoluotain decides its plate position because a 2-D shadow has a shape. The corner camera (Bouman et al. 2017) gives the same in 2-D: on the floor beside a wall edge, extra light from the hidden side depends on the angle around the corner, so iso-intensity lines are rays from the corner.

**Why no camera calibration is needed.** A camera maps the floor plane to the image by a homography. Homographies map lines to lines, so the rays stay lines through one image point, the corner's image. Perspective only remaps the angle monotonically, and free angular radiance bins absorb that remapping (`test_homography_maps_floor_rays_to_image_lines_through_the_apex`). Locating the corner is therefore a 2-parameter search, like Varjoluotain's plate search.

## Method (frozen at this commit)

Code: `src/sighextraimage/fan.py`, benchmark `src/sighextraimage/gate3.py`.

**Measurement.** Mean linear light (not log: hidden light adds to room light) in 8 × 8 or 16 × 16 pixel blocks of the floor mask. Noise per channel comes from the MAD of the 5-point Laplacian of the block image (counts albedo texture as noise). The correlation length comes from the Laplacian's lag-1 autocorrelation. A quadratic background (constant, x, y, x², y², xy) is projected out of data and kernels.

**Model-resolution floor.** σ_eff² = σ² + (0.20 · rms of projected data)². Development showed that the true geometry leaves 0.03–0.3% of variance unexplained even noise-free (finite bins and radial terms). At high signal-to-noise, that approximation error otherwise turns into confident false refutations. 0.20 was sized from the measured approximation residue.

**Hypothesis.** `(apex x, apex y, hidden side, penumbra)`. Kernel: for each block, averaged over 4 × 4 sub-points, a soft step in image angle about the apex at each of `n_bins` positions tiling the observed span. Each step is multiplied by two non-negative radial hats in log image-radius. Each colour channel is fitted by NNLS.

**Funnel.**
- Level 1: blocks 16, 24 bins, apex grid step 16 px over the image ± one image size, penumbra 0.03, cap 600.
- Level 2: blocks 8, 48 bins, step 4 px, children of survivors, cap 200.
- Level 3: blocks 8, 48 bins, step 2 px, penumbra ∈ {0.01, 0.03, 0.08}, cap 400.
- Survival: `chi2 ≤ chi2_best + Q99(chi2, 3) · max(1, Birge) · c`.

**Signal test.** Background-only is nested in every fan. The background excess over the best fan, in units of `Q99(chi2, 4 + 3·2·n_bins) · max(1, Birge) · c`, must exceed **3.0**. The nominal value of 1.0 gave false alarms on 4 of 12 development null scenes (maximum 1.75); every development signal scene exceeded 9.

**Verdicts.**
- `no-signal`
- `refuted-model` (Birge > 4)
- `located` / `located-side-undecided`: the survivor cloud's long axis ≤ 16 px
- `on-a-line`: long axis > 16 px, short axis ≤ 16 px
- `undecided` otherwise

## Synthetic world (frozen)

`render_fan_scene(seed, leak=0.3, texture=0.01, noise_std=0.002)`, 256 × 256, 8-bit.
- **Floor:** corner at the floor origin, wall on φ ≤ 0.
- **Camera:** pinhole 2–3.5 m from the corner, height 1–1.7 m, random roll, looking at the floor near the corner. Wall pixels are masked.
- **Hidden sources:** 1–3, at hidden angle 0.15–1.35 rad, each a soft edge of width 0.03–0.15, distance 1.5–3.5, random colour.
- **Room light:** ambient 0.35 with a ±25% linear gradient.
- **Floor:** albedo texture with log-std 0.01 at 1.5–4 px scale.

The leak is scaled so the peak hidden light is 30% of ambient. After background removal, development measured the fan structure at 3–7× the block-level texture for this setting.

Null scenes: the same seed with the hidden light removed.

**Edges.** Sources with hidden angles ≥ 0.25 rad apart cast distinct shadow lines (`distinct_edges`). With one distinct edge, the corner is only constrained to lie on that edge's line.

Held-out seeds: `41 43 47 53 59 61 67 71 73 79 83 89 97 101 103 107`.

## Pass criteria

| ID | Claim | Criterion |
|---|---|---|
| G3a | No false alarms | at most 1 of 16 null scenes not `no-signal` |
| G3b | Two or more distinct edges → corner located | ≥ 75% of those scenes `located*` within 8 px (one block) |
| G3c | One distinct edge → corner on the reported line | ≥ 75% detected and true corner within 16 px of the survivor line |
| G3d | Side right when located | hidden side correct in ≥ 90% of `located*` verdicts |
| G3e | Never confidently wrong | no `located*` verdict more than 16 px from the true corner |

Gate 3 passes only if all five pass.

## Preregistered predictions (not criteria)

- With one distinct edge, the hidden side is often not decided correctly: the flipped orientation also fits a single edge, as in 1-D.
- The sweep settings `(leak 0.1, texture 0.01)` and `(leak 0.3, texture 0.03)` put the fan structure near or below the floor texture (measured ratio 0.4–2.3 in development). Localisation should degrade there. The safety claim G3e should still hold.

## Claim boundary

Synthetic floor, ideal Lambertian corner-camera physics, known floor mask. A pass would mean the bench can locate a corner from the light on the floor when the fan structure exceeds the floor texture, and says so when it can't. It does not mean an arbitrary photo has a usable fan: real floors have stronger texture and real hidden scenes are often dimmer than this setting.
