# Gate 2 — corner geometry funnel and calibrated temperature

Date: 2026-10-03
Status: preregistered before any held-out seed was run. Development used only the
Gate 0–1 seeds (3, 7, 11, 19, 23).

## Why this gate exists

The Gates 0–1 v0 receipt left two open problems.

1. **ESS collapse on extracted pixels.** The affine residual is a *fraction of
   variance* (between 0 and 1), but it was divided by `2·sigma²` with `sigma = 0.01`
   as though it were an intensity. Fraction gaps of 0.005 became log-weight gaps of
   20–500, so the effective sample size collapsed to 1 for correct *and* mirrored physics.
2. **The corner geometry is assumed, never checked.** Photo mode uses one fixed
   corner geometry for every photograph.

Development also found a third issue, recorded here before the gate is run:

3. **The mirrored control is a relabeling, not a competing physics.**
   `WrongCornerTransport` at hidden angle `theta` predicts the same boundary as
   `CornerTransport` at `1.30 − theta`; the measured difference is 4e-7 relative.
   No measurement can prefer one over the other. Gate 0's "correct localises far
   better than mirrored" was scored against true theta, so it measured the
   labeling convention, not evidence. Gate 2 drops this control and uses
   hypotheses the data can refute.

## Method (frozen at this commit)

Code: `src/sighextraimage/geometry_funnel.py`, benchmark `src/sighextraimage/gate2.py`.

**Hypotheses.** A corner geometry is `(phi_start, phi_end, penumbra, reversed)`: the
hidden angles the selected strip spans, the softness of the edge's shadow, and
whether the edge sits at the other end of the strip. Search box:
`phi_start ∈ [0, 0.45]`, `phi_end ∈ [0.75, 1.30]`, `penumbra ∈ [0.02, 0.16]`
(log grid), `phi_end − phi_start > 0.2`, both orientations.

**Funnel (after Varjoluotain's occluder funnel).** For each hypothesis, the best
non-negative hidden radiance is fitted (NNLS, each colour channel free) after a
constant and a ramp are projected out of the profile and the kernel. The free
radiance bins tile the hypothesis's own span (radiance below the span is a
constant on the strip; radiance above it reaches no sample), so no geometry wins
by discretisation. Level 1: 32 samples × 12 bins, grid step (0.05, 0.05, 0.35 in
log penumbra). Level 2: 64 × 16, half the step, children of survivors only. A
hypothesis survives if `chi2 ≤ chi2_best + Q99(chi2, k=3) · max(1, Birge) · c`,
where `c` is the noise correlation length. Refinement stops when a level refutes
less than half of what it tested.

**Noise from the photo.** Rows of the strip are split into two interleaved halves
(alternating blocks of 4 rows). Each half goes through the same extraction; half
their difference, with constant + ramp removed, gives a per-channel sigma (max of
std and 1.4826·MAD) and an integrated autocorrelation length `c`.

**Signal test.** Background-only (constant + ramp) is nested inside every corner
hypothesis. It is refuted when its chi2 exceeds the best corner fit by more than
`Q99(chi2, k = 3 + 3·n_bins) · max(1, Birge) · c`.

**Verdicts.** `refuted-model` (best Birge > 4), `no-signal` (background not
refuted), `decided-geometry` (reversed edge refuted and ≥ 90% of the coarse box
refuted), otherwise `undecided-geometry`.

**Calibrated posterior.** Gaussian likelihood in measurement units using the
photo's own sigma; one shared non-negative exposure gain; constant + ramp removed
per channel; chi2 divided by `c`; temperature = Birge ratio of the best
(candidate, geometry) pair, floored at 1. The hidden-scene posterior is
marginalised over a geometry set: 48 draws from the geometry prior (seeds
777–824), minus those the funnel refutes. Sampling is population Monte Carlo:
2000 prior draws, then 5 rounds of 2000 from a defensive mixture (20% prior, 80%
Gaussians around 64 resampled particles), reweighted by the deterministic-mixture
rule so the importance weights are exact.

## Synthetic world (frozen)

Scene `72 × 96`, 64 hidden columns, latent prior as in Gates 0–1. For seed `s`:
latent from `sample_prior(1, seed=s)`; true geometry from
`numpy.random.default_rng(s + 5000)`: `phi_start ~ U(0.02, 0.40)`,
`phi_end ~ U(0.85, 1.30)`, `log penumbra ~ U(log 0.025, log 0.12)`, never reversed.
Leakage rendered by `render_visible_with_leakage(noise_std=0.002, texture_seed=s+91)`,
then quantised to 8 bits. Null scenes use the same seed with hidden brightness 0.
Initial candidate pool: `sample_prior(2000, seed=s+10000)`.

## Held-out seeds

`41 43 47 53 59 61 67 71 73 79 83 89 97 101 103 107` (16 signal scenes, 16 null scenes).

## Pass criteria

| ID | Claim | Criterion |
|---|---|---|
| G2a | No false alarms | background-only refuted in at most 1 of 16 null scenes |
| G2b | The truth is never refuted | the true geometry's cell survives every funnel level in at least 15 of 16 signal scenes |
| G2c | Calibrated uncertainty | the 90% theta interval of the prior-marginalised posterior contains true theta in at least 12 of 16 signal scenes |
| G2d | No collapse | median ESS of the prior-marginalised posterior at least 20 |

Gate 2 passes only if all four pass.

## Preregistered predictions (not pass criteria)

- The funnel will **not** decide the geometry: median refuted share of the coarse
  box below 0.5, and the reversed edge refuted in fewer than half of the signal
  scenes. A 1-D boundary profile is a cumulative integral of the hidden radiance,
  so remapping the strip's angle axis is largely absorbed by the radiance.
- The default fixed geometry with the calibrated temperature will cover true
  theta less often than the prior-marginalised posterior.
- Scenes whose hidden object lies outside the strip's span will read `no-signal`.

## Also reported (descriptive)

Detection rate on signal scenes; the v0 posterior's ESS on the same seeds
(`affine`, `sigma = 0.01`, default geometry); default-geometry and oracle-geometry
posteriors (coverage, theta error, ESS); prior baseline theta error; temperatures;
noise sigma and correlation; full funnel ledgers.

## Claim boundary

A pass means: inside this synthetic world, the bench knows when a boundary carries
an occluder signal, never rules out the true geometry, and reports hidden-angle
uncertainty that is calibrated when the geometry prior matches the world. It does
not mean a real photograph decides its corner geometry, and the geometry prior for
a real photo is an assumption the user supplies.
