# Gate 4 — certify before claiming: the corner is a point only if probing finds no other

Date: 2026-10-04
Status: preregistered before any held-out result was computed. Fresh held-out seeds. Development used the Gate 3 dev seeds (2–37) and the Gate 3 held-out seeds (41–107), which became development data once their results were published.

## Why

Gate 3 located the corner within 8 px in 6 of 7 scenes with two or more shadow edges. But two single-edge scenes were reported `located` 238 and 282 px off, and the low-signal sweep added four more confident errors. Diagnosis on those scenes:

1. **One edge pins the corner only to a half-line.** Moving the hypothesised corner along the edge's image line, *away* from the visible floor, keeps the edge consistent. Moving it toward the floor is refuted at once. In every Gate 3 scene, the best-fit apex lay on a true edge line (0.04–18 px), even when it was 200–400 px from the corner.
2. **That valley is 1–2 px wide.** The cost across it rises by about 2000 per 16 px. Gate 3's grid (16 → 4 → 2 px) and any 15° probe ring step over it. The survivors were a short fragment, which read as "compact", which read as "located".
3. **More radial freedom does not help.** With 3, 4 or 6 radial terms, or a positive-plane amplitude model, the slope along the valley remains (seed 89).

So the fix is not a better model. It is to **check before claiming**.

## Method (frozen at this commit)

Code: `certify()` in `src/sighextraimage/fan.py`, benchmark `src/sighextraimage/gate4.py`. The Gate 3 funnel and its detection test (threshold 3.0) are unchanged. Certification runs on every scene the funnel detects:

1. **Re-check, finest model.** Block 8 px, 48 angular bins, penumbra ∈ {0.01, 0.03, 0.08}, same noise floor and survival rule as Gate 3. Evaluate:
   - the funnel's survivors;
   - the best apex with both hidden sides;
   - a coarse ring of 24 directions × radii {16, 32, 64, 128} px, both sides.
2. **Find the valley.** On a fine ring at 24 px (72 directions, then 0.5° refinement), take for each line direction the cheaper of its two antipodal points, because a single edge leaves a half-line.
3. **Walk the valley** both ways from the best apex in 12 px steps, up to 512 px. At each step, take the best of 5 lateral offsets (−4…+4 px) and keep going until the photo refutes the point.
4. **Verdict from all evaluated positions within the survival threshold:**
   - `located`: long axis ≤ 16 px. `located-side-undecided` if both hidden sides survive.
   - `on-a-line`: short axis ≤ 16 px, or long ≥ 4 × short with short ≤ 32 px. The reported line passes through the best apex along the survivors' main axis when they span > 64 px, otherwise along the fine valley direction.
   - `undecided` otherwise.

## Held-out seeds

`109 113 127 131 137 139 149 151 157 163 167 173 179 181 191 193`

These cover the primary setting (`leak 0.3, texture 0.01`): 16 signal scenes plus 16 null scenes. The two sweep settings, `leak 0.1, texture 0.01` and `leak 0.3, texture 0.03`, run on the same 16 seeds.

## Pass criteria

| ID | Claim | Criterion |
|---|---|---|
| G4a | No false alarms | at most 1 of 16 null scenes detected |
| G4b | ≥ 2 distinct edges → located | ≥ 75% of those scenes `located*` within 8 px |
| G4c | One edge → the line, never a point | ≥ 60% of single-edge scenes detected, not `located*`, with the true corner within 16 px of the reported line |
| G4d | Side right when located | ≥ 90% of `located*` verdicts |
| G4e | **Never confidently wrong** | zero `located*` verdicts more than 16 px off across all 48 signal scenes (primary + both sweeps) and all null scenes |

Gate 4 passes only if all five pass. G4e is the gate's purpose.

## Development estimates (stated so the result can be read against them)

On the 14 Gate 3 held-out scenes re-run with certification:
- two-edge scenes located within 6 px: 5/6;
- single-edge scenes located: 0/8;
- single-edge line within 16 px: 5/8 (62%) — G4c is set just below this, not at a comfortable margin;
- confident errors: 0.

On 16 low-signal development scenes: 0 confident errors (Gate 3 had 4 in its held-out sweep).

## Claim boundary

Synthetic floor, Lambertian corner-camera physics, known floor mask. A pass would mean the bench reports a corner as a point only when the photo rules out every other position it checked. That includes the single-edge valley a coarse search misses. It does not mean every possible alternative was checked: positions outside the funnel's box and the walk's 512 px reach are not tested.
