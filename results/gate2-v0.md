# Gate 2 v0 — corner-geometry funnel + calibrated temperature

**Result: Gate 2 fails, narrowly, on two of four preregistered criteria.** The protocol was committed (d98253b) before any held-out seed ran: [`docs/superpowers/specs/2026-10-03-gate2-geometry-funnel-design.md`](../docs/superpowers/specs/2026-10-03-gate2-geometry-funnel-design.md). Receipt: [`receipts/gate2-v0.json`](receipts/gate2-v0.json). Seeds 41–107 (16 signal scenes + 16 null scenes).

| ID | Claim | Observed | Required | |
|---|---|---:|---:|---|
| G2a | No false alarms on null scenes | 2/16 | ≤ 1 | **fail** |
| G2b | True geometry never refuted | 16/16 | ≥ 15 | pass |
| G2c | 90% θ interval covers truth | 11/16 | ≥ 12 | **fail** |
| G2d | No ESS collapse (median) | 48 | ≥ 20 | pass |

## What the numbers say

**The v0 collapse is fixed.** On the same held-out scenes the v0 posterior (affine fraction residual ÷ 2·0.01²) has median ESS **1.0**; the calibrated posterior has median ESS **48**. The cause was a unit mismatch: the affine residual is a fraction of variance, and σ = 0.01 was applied to it as if it were an intensity.

**A single boundary photo does not decide the corner geometry** (preregistered prediction, confirmed). All 16 signal scenes read `undecided-geometry`. Median refuted share of the search box: **3.6%**. Reversed edge refuted: **0/16**. A 1-D boundary profile is a cumulative integral of the hidden radiance. Remapping the strip's angle axis, or even flipping it, is absorbed by a different non-negative radiance plus the removed ramp (see `test_nonnegativity_barely_constrains_a_one_dimensional_profile`).

**What the photo does decide: whether there is an occluder signal at all.** 16/16 signal scenes detected. 14/16 null scenes correctly read `no-signal`. The two false alarms sit just over threshold (1.04× and 1.34×).

**Assuming the default geometry is the dangerous choice:**

| Posterior | 90% coverage | median θ error | median ESS |
|---|---:|---:|---:|
| prior only (no data) | — | 0.278 | — |
| v0 (fraction residual, default geometry) | — | 0.085 | 1.0 |
| calibrated, **default geometry** | **2/16** | 0.092 | 13 |
| calibrated, marginalised over geometry prior | 11/16 | 0.062 | 48 |
| calibrated, oracle (true) geometry | 14/16 | 0.006 | 29 |

With the true geometry, θ is pinned to 0.006 rad. With an assumed geometry, the answer is confidently wrong 14 times in 16. Marginalising over the geometry prior is honest about the spread and gets the best error without the oracle.

**Why G2c fails.** Four of the five misses have small ESS (1, 2, 9, 24) and a collapsed interval: the importance pool locked onto one geometry mode of a multi-modal posterior (seed 103: interval [0.56, 0.56], ESS 1). The fifth (seed 59) misses by 0.01 at the interval edge. This points at the sampler (one pool shared by 48 geometry hypotheses), not at the likelihood. The same likelihood with the true geometry covers 14/16. This is a diagnosis, not a pass: fixing it is a new gate.

## Next gate, if pursued (not run)

- **Gate 2b:** per-geometry stratified sampling, so each surviving geometry gets its own refined pool and the θ posterior is a proper geometry mixture; same seeds' *protocol*, fresh held-out seeds.
- **Real geometry from image geometry, not light.** Bouman et al.'s corner camera and Seidel et al.'s 2-D single-edge method take the corner location as known and compute each floor pixel's angle from where it sits relative to the corner. In 2-D the penumbra is a fan whose apex is the corner. Locating that apex is the true analogue of Varjoluotain's plate search. The current renderer collapses rows, so it has no fan to find.

## Claim boundary

Synthetic world only. The geometry prior used for marginalisation is the generator's own prior. For a real photo, that prior is an assumption the user supplies, and the funnel shows the photo will not narrow it much.
