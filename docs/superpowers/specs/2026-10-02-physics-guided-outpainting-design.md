# SighExtraImage — Physics-Guided Outpainting Design

**Date:** 2026-10-02  
**Status:** revised after GUI/control design review; ready for written-spec approval

## Goal

Build a scientifically honest path from weak optical evidence at an image boundary to constrained hypotheses about what lies outside the frame, while keeping the project usable as a local photo-inspection tool.

The core claim to test is narrow:

> Weak in-frame optical leakage can reduce uncertainty about an out-of-frame physical latent, and a correct likelihood can constrain a shared prior more than a wrong or uninformative likelihood.

The project must never equate a plausible generated continuation with a measured reconstruction.

## Design principle

One scientific core serves both the benchmark and the GUI.

There is no separate “demo physics.” Synthetic experiments, CLI commands, the GUI, later diffusion guidance, and eventual Varjoluotain integration must call the same transport, extraction, likelihood, inversion, posterior, metrics, and receipt code.

## Core representation

Represent the unseen region first as a low-dimensional physical latent rather than raw RGB pixels:

```text
z_physical = {
  angular / image-plane position,
  size / extent,
  shape,
  albedo / coarse RGB,
  occupancy,
  coarse layout / depth,
  light parameters
}
```

The observed boundary signal is

```text
y_R = A_R(z_physical) + noise
```

where `R` is an explicit visible measurement region and `A_R` is a cheap differentiable proxy for indirect light transport / penumbra formation onto that region.

This separates:

- **physics:** hidden latent -> predicted boundary evidence;
- **appearance prior:** hidden latent / hidden pixels -> plausible scene appearance.

A later diffusion model may provide natural-image plausibility, but the physics term is allowed to constrain only what the observed region can support.

## Scientific gates

### Gate 0 — Does the boundary signal carry recoverable information?

Using the same candidate set from a declared prior, compare:

1. **prior-only** — uniform prior weights;
2. **correct physics** — weights from `p(y | z, A)`;
3. **wrong physics** — same hypotheses, mismatched transport;
4. **no-occluder ablation** — transport with angular visibility structure removed or collapsed toward an all-visible/rank-poor operator.

A positive result means only that the declared synthetic measurement narrows some hidden attributes.

### Gate 1 — Does the physics gradient point toward truth?

Before any diffusion model, test the actual likelihood gradient.

For wrong initial hypotheses, evaluate

```text
L(z) = residual(y_R, A_R(z))
```

and test whether `-grad_z L` moves recoverable continuous latent attributes toward truth better than:

- wrong-physics gradients;
- norm-matched random directions;
- no-occluder gradients.

This gate prevents “differentiable” from being mistaken for “useful guidance.”

### Gate 2 — Posterior contraction and ambiguity structure

Quantify how much the likelihood narrows the physical latent and which attributes remain degenerate. The output should be a family of compatible hidden states, not one authoritative reconstruction.

### Gate 3 — Generative / diffusion guidance

Only after Gates 0–2 survive, attach a pretrained outpainting prior.

Conceptual loop:

```text
x_t
  -> denoiser predicts x0_hat
  -> hidden representation / proxy
  -> A_R(...) predicts y_hat
  -> residual(y_R, y_hat)
  -> likelihood guidance nudges reverse diffusion
```

The diffusion prior supplies image plausibility; boundary evidence supplies a limited physical constraint.

### Gate 4 — Real Varjoluotain measurement extraction

Replace perfect synthetic `y` with a descriptor extracted from real visible pixels:

```text
y_hat = extract_boundary_signal(I_visible, R)
```

Only this stage tests whether real-pixel illumination leakage can constrain the hidden-scene posterior.

## Synthetic scene model

Use a simple hidden world outside the crop, with at minimum:

- position / angular position;
- size / angular width;
- RGB albedo / radiance;
- shape class (`disk`, `rectangle` initially);
- occupancy;
- one or two simple lighting parameters.

The hidden object must never enter the visible crop.

Synthetic generation saves separately:

- full ground-truth scene for evaluation only;
- camera-visible crop;
- hidden latent truth;
- boundary region mask `R`;
- noiseless physical boundary signal;
- noisy boundary signal;
- all nuisance and transport parameters.

## Boundary measurement region R

The likelihood operates only on pixels or derived samples the camera actually observes.

For v0, `R` is a narrow user- or program-defined wall/floor/corner strip adjacent to the crop boundary. The operator predicts a compact `[M,3]` boundary profile rather than the whole panorama.

This supports the distinction:

> Structure stable across posterior samples may be measurement-supported. Fine details that vary across samples remain prior-filled imagination.

## Forward transport

Do not begin with a path tracer.

Use an auditable differentiable corner/penumbra operator with:

- soft angular visibility from an occluding edge;
- smooth distance falloff;
- coarse color transport;
- optional soft penumbra;
- ambient / gain nuisance terms.

A useful v0 form is a smoothed visibility/transport matrix:

```text
y = K(geometry) @ L_hidden + ambient
```

where the occluder makes rows of `K` observe different hidden angular wedges. The no-occluder ablation replaces this structured visibility with an all-visible or otherwise deliberately information-poor operator.

The transport's condition number / singular spectrum should be inspectable, because inverse ill-conditioning is part of the scientific question rather than a hidden implementation detail.

## Physics-only inversion baseline

Add a classical inversion baseline with no learned semantic prior.

Solve a constrained/regularized problem such as

```text
min_L residual(y, K @ L) + lambda_tv * TV(L)
subject to L >= 0
```

This baseline asks how much angular/color structure the photons alone support before a generative prior is introduced.

It must be reported separately from posterior inference and later diffusion results.

## Likelihoods and nuisance invariance

### Exact synthetic likelihood

When the synthetic renderer and observation model share calibrated radiometry, allow direct Gaussian/L2 likelihood:

```text
log p(y | z) = -|| y - A(z) ||^2 / (2 sigma^2)
```

### Nuisance-invariant likelihood

Real photographs contain unknown exposure, ambient illumination, white balance, wall albedo, tone mapping, and camera response. For photo mode, raw radiance error must not be assumed meaningful.

Provide at least:

1. `l2` — calibrated synthetic use only;
2. `affine` — fit one shared gain plus per-channel offsets before residual measurement;
3. optionally `affine_per_channel` — diagnostic color-blind ablation.

The preferred real-photo residual is conceptually:

```text
min_{a,b_r,b_g,b_b} || y - (a * y_hat + b) ||^2
```

with `a` shared across channels to preserve relative chromatic structure.

## Boundary-signal extraction

Add an explicit extraction module between pixels and the inverse model.

### Synthetic extraction check

Bake the true leakage into a textured visible surface, then extract it back from the rendered crop. Report correlation / shape agreement between extracted and true signal.

This isolates the extraction problem from the inverse problem.

### Still-image extraction

Initial still-image extractor may use:

- linearized RGB where possible;
- log-domain decomposition;
- averaging across the selected strip;
- low-pass smoothing along the boundary;
- detrending / nuisance normalization;
- chromatic and derivative profiles.

It must be described as an estimator, not a ground-truth measurement.

### Video extraction

Reserve a temporal extractor in which static albedo largely cancels under frame differences / robust temporal baselines. Video may become the stronger real-world mode if still-image albedo/illumination separation proves underdetermined.

## Gate 0 inference

Use one shared candidate batch for every arm.

For candidate `z_i`:

```text
E_i = residual(y, A(z_i))
log w_i = -E_i / temperature_or_noise_scale
```

Normalize with log-sum-exp.

Report:

- posterior mean / median where meaningful;
- MAP candidate only as a summary;
- position/size/color error;
- shape probability;
- posterior entropy / credible width;
- effective sample size;
- contraction relative to prior;
- correct vs wrong vs no-occluder performance.

## Gate 1 gradient controls

For continuous latent coordinates:

- compare autograd against finite differences;
- normalize latent coordinates by declared prior ranges before truth-distance comparisons;
- record componentwise direction agreement;
- measure error before/after one step and a short optimization trajectory;
- compare correct physics, wrong physics, no occluder, and norm-matched random controls.

A loss decrease is not sufficient if latent truth error gets worse.

## GUI / local application

Use **Gradio** as a thin local browser front end.

Launch target:

```bash
sighextraimage gui
```

The GUI must call the same core functions used by CLI/scientific tests.

### GUI Mode A — Synthetic Lab

Purpose: make the falsifiable mechanism visible and easy to explore.

Controls:

- hidden position / angle;
- size / width;
- RGB/albedo;
- shape;
- noise level;
- occluder on/off;
- penumbra / transport softness;
- likelihood mode;
- candidate count / posterior temperature within safe bounded defaults.

Displays:

- full hidden truth (evaluation view only);
- camera-visible crop;
- selected boundary region `R`;
- true and extracted boundary profiles;
- transport kernel / singular-value or condition diagnostic;
- physics-only TV inversion;
- prior vs correct-physics vs wrong-physics vs no-occluder posterior summaries;
- posterior contraction / uncertainty plots;
- Gate 1 gradient direction / error-change diagnostics.

The GUI must clearly mark ground-truth panels as synthetic-only information unavailable for real photos.

### GUI Mode B — Photo Inspector

Purpose: let users test whether a real photo contains a potentially informative boundary signal before any generative reconstruction is attempted.

Inputs:

- upload JPG/PNG;
- choose crop edge / corner orientation;
- choose or draw measurement region `R`;
- choose smoothing / extraction mode;
- choose likelihood nuisance mode.

Outputs:

- original photo with `R` overlay;
- extracted low-frequency illumination profile;
- derivative/chromatic diagnostics;
- estimated signal strength / SNR proxy;
- transport conditioning for the selected geometric model;
- optional physics-only inverse profile;
- warnings when evidence is weak, geometry is incompatible, or the inference is underdetermined.

V0 Photo Inspector must **not** present a generated hidden panorama as measured truth.

Recommended wording when the signal is not informative:

> No evidence that this selected boundary strongly constrains the unseen region under the current model.

### Later GUI Mode C — Constrained Outpainting

Only after Gate 3 exists, add side-by-side galleries:

- ordinary prior-only outpainting;
- physics-guided outpainting;
- multiple samples for each arm;
- boundary re-rendering residual;
- cross-sample stability / uncertainty.

Never show only one “revealed” image.

## Software architecture

```text
src/sighextraimage/
  latent.py          # physical latent z and parameter transforms
  scene.py           # synthetic scene/crop rendering
  transport.py       # corner/penumbra A(z), wrong/no-occluder controls
  extraction.py      # synthetic/still/video boundary-signal extraction
  inversion.py       # physics-only TV inversion
  likelihood.py      # l2 / affine nuisance-invariant residuals
  prior.py           # declared latent prior
  inference.py       # Gate 0 posterior weighting/summaries
  gradient_gate.py   # Gate 1 guidance falsifier
  metrics.py         # errors, uncertainty, conditioning, extraction metrics
  benchmark.py       # deterministic scientific runs
  receipts.py        # JSON receipts
  gui.py             # thin Gradio adapter over core functions
  cli.py             # CLI including `gui`

tests/
  test_latent.py
  test_scene.py
  test_transport.py
  test_extraction.py
  test_inversion.py
  test_likelihood.py
  test_inference.py
  test_gradient_gate.py
  test_gui_smoke.py
  test_benchmark.py

results/
  receipts/
```

## Numerical stack

- Python 3.11+
- PyTorch CPU for differentiable physics and Gate 1
- NumPy where convenient
- Pillow for ordinary image I/O
- Gradio for local GUI
- matplotlib optional for saved diagnostics; GUI plots may use standard supported plotting components
- pytest for verification

No pretrained diffusion dependency in the first Gates 0–1 implementation.

## Interfaces reserved for later diffusion work

Core physics/extraction API should expose functions equivalent to:

```text
sample_prior(n, rng) -> z candidates
render_hidden(z) -> hidden representation
predict_boundary(z, scene_context) -> y_hat
extract_boundary_signal(image_or_frames, region, config) -> y
physics_residual(y, y_hat, mode) -> scalar
physics_loss(z, y, scene_context, mode) -> scalar
```

A later generative adapter may add:

```text
denoise(x_t, t, known_crop) -> x0_hat
predict_boundary_from_hidden_image(x0_hat_hidden, scene_context) -> y_hat
```

without changing benchmark definitions.

## Outputs and receipts

Scientific runs save:

- seed/config metadata;
- transport and extraction parameters;
- condition numbers / diagnostic spectra where relevant;
- Gate 0 per-arm metrics;
- TV-inversion metrics;
- no-occluder ablation;
- Gate 1 componentwise diagnostics;
- instability/non-finite counts;
- exact command/config used.

GUI exploratory runs may optionally export a receipt, but must not silently enter the scientific benchmark dataset.

## Predeclared first-milestone criteria

### Gate 0

On fixed synthetic seeds, correct physics must outperform prior-only on at least one recoverable hidden attribute and outperform wrong physics on that same attribute. Posterior uncertainty must contract for at least one continuous attribute.

The no-occluder ablation should lose angular/position information relative to the corner geometry. If it does not, the interpretation must be narrowed before claiming computational periscopy.

### Gate 1

Correct-physics gradients must show truth-directed/error-reducing behavior above wrong/no-occluder/random controls for at least one identifiable continuous attribute. Null or misleading components are reported explicitly.

### Extraction

The synthetic pixel-extraction path must be scored against the known baked-in signal before real-photo inference is interpreted. Poor extraction narrows the claim to idealized measurements even if Gates 0–1 succeed.

### GUI

The local GUI must launch without model downloads, run Synthetic Lab entirely on CPU, accept ordinary JPG/PNG files in Photo Inspector, and clearly distinguish measurement diagnostics from generated/inferred hidden content.

## Claim boundary

A positive early result means:

> In the declared synthetic setup, weak visible boundary light contains recoverable information about some out-of-frame physical attributes; structured occlusion contributes information beyond a no-occluder control; and the specified likelihood can narrow a shared prior and/or provide useful local guidance.

It does **not** mean:

- arbitrary still photographs reveal unique hidden scenes;
- the proxy is a complete real-room light transport model;
- smooth image gradients are necessarily hidden-object evidence;
- a generated continuation is ground truth;
- diffusion success has been demonstrated before Gate 3;
- the GUI is a “see around corners” device merely because it produces a visualization.

## First implementation milestone

The first useful repository state is complete when:

1. package + CLI + `sighextraimage gui` launch locally;
2. Synthetic Lab exercises the same code as the benchmark;
3. Gate 0 includes prior/correct/wrong/no-occluder arms;
4. physics-only TV inversion is available;
5. L2 and affine-invariant likelihoods are implemented/tested;
6. Gate 1 tests correct/wrong/no-occluder/random gradients;
7. synthetic boundary-signal extraction is scored against known truth;
8. Photo Inspector accepts JPG/PNG and displays measurement diagnostics without making a hidden-scene claim;
9. deterministic multi-seed JSON receipt records the first scientific outcome;
10. README reports the result whether positive, partial, or negative.

## Roadmap

```text
Gate 0: information in ideal boundary signal
  -> Gate 1: useful physics gradient
  -> synthetic pixel extraction
  -> Gate 2: ambiguity / contraction map
  -> real-photo / video measurement characterization
  -> Gate 3: diffusion-guided outpainting
  -> Gate 4: Varjoluotain-integrated real measurements
```

The guiding distinction remains:

> Plain outpainting asks what could plausibly be outside the crop. SighExtraImage asks which plausible hidden worlds remain compatible with the photons that leaked into the visible boundary.
