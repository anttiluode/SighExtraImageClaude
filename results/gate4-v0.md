# Gate 4 v0 — certify before claiming

**Result: Gate 4 fails on one scene.** Seed 179 is the only located verdict that is wrong. It fails G4e (one confident error) and G4d (its hidden side is flipped). The other three criteria pass. The same scene also points at the missing check: a fit that "explains" the light by cancelling huge contributions against each other.

Protocol, committed (bad39f7) before any held-out result: [`docs/superpowers/specs/2026-10-04-gate4-certified-apex-design.md`](../docs/superpowers/specs/2026-10-04-gate4-certified-apex-design.md). Receipt: [`receipts/gate4-v0.json`](receipts/gate4-v0.json). Fresh seeds 109–193: 16 signal and 16 null scenes, plus 32 low-signal sweep scenes.

| ID | Claim | Observed | Required | |
|---|---|---:|---:|---|
| G4a | No false alarms | 0/16 | ≤ 1 | pass |
| G4b | ≥ 2 distinct edges → located within 8 px | **6/6** | ≥ 75% | pass |
| G4c | One edge → a line, never a point | 6/10 | ≥ 60% | pass |
| G4d | Side right when located | 7/8 | ≥ 90% | **fail** |
| G4e | Never confidently wrong | 1 of 10 located | 0 | **fail** |

## What certification fixed

| | Gate 3 | Gate 4 |
|---|---|---|
| Two-edge scenes located within 8 px | 6/7 | **6/6** (errors 0.6–7.6 px) |
| Confident errors, primary setting | 2 | 1 |
| Confident errors, low-signal sweeps | 4 | **0** of 32 scenes |
| Single-edge line within 16 px | 3/9 | 6/10 |

Of the 10 located verdicts across all 48 signal scenes, 9 are within 7.6 px of the true corner. Single-edge scenes now report the edge's line, and for 7 of the 10 the line passes within 6 px of the corner.

## The failure: seed 179

Seed 179 has one hidden source, so one shadow edge. The best fit sits on that edge's image line (3.4 px off it) but at the *far* end of it, 348 px from the corner, with the hidden side flipped. There it genuinely fits better than the truth: Δχ² = 87 against a survival threshold of 11.

The valley along the edge line rises by 34 within 24 px, so the walk stopped immediately and every probe was refuted. Probing cannot catch a valley that is both narrow and steep.

## What separates it (diagnostic on held-out data, so evidence for Gate 5, not a fix)

For each certified scene, I compared the energy of the fitted angular bins (after background removal) with the energy of the data they explain. A physical fan adds light; it does not need large contributions that cancel each other.

| Verdict | Σ bin energy / data energy |
|---|---|
| 9 correct located verdicts | **0.23–0.86** |
| Seed 179 (located, 348 px off) | **35.7** |
| Positions > 140 px from the corner (all settings) | 27.7–1962 |
| Positions 78–96 px off | 1.25–4.1 |

A ceiling of about 2 would have demoted seed 179 and kept all nine correct verdicts. I picked that value after seeing the held-out data, so it can't count as a result here. It becomes the next gate's hypothesis, to be run on fresh seeds.

## Next gate (not run)

**Gate 5:** keep Gate 4 and add a plausibility requirement for `located`: Σ bin energy ≤ 2 × data energy, plus the edge count for the record. Freeze, then test on fresh seeds, including scenes made to provoke the mirror case: single edge with the corner inside the frame.

## Claim boundary

Synthetic floor, Lambertian corner-camera physics, known floor mask. "Never confidently wrong" was tested on 48 signal scenes and 16 null scenes, and failed once.
