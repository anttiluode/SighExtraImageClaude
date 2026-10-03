# SighExtraImage Gates 0–1 Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the scientific core that tests whether weak boundary light constrains an out-of-frame physical latent, with explicit no-occluder, wrong-physics, physics-only inversion, extraction, and gradient-direction controls.

**Architecture:** A low-dimensional `PhysicalLatent` is rendered into a hidden scene and mapped through a differentiable corner/penumbra transport operator into an observed boundary profile `y_R`. The same core supports synthetic ground truth, pixel-level extraction, classical TV inversion, prior/posterior inference, Gate 1 gradients, and later GUI/diffusion adapters.

**Tech Stack:** Python 3.11+, PyTorch CPU, NumPy, Pillow, pytest, standard-library JSON/argparse/pathlib/dataclasses; matplotlib optional for saved diagnostics.

**Spec:** `docs/superpowers/specs/2026-10-02-physics-guided-outpainting-design.md`

## Global Constraints

- No pretrained diffusion model or external model weights in Gates 0–1.
- Hidden geometry remains outside the visible crop.
- Physics residual is evaluated only on explicit observed region `R`.
- Prior-only, correct-physics, wrong-physics, and no-occluder arms share the same candidate hypotheses.
- Transport is deterministic before declared measurement noise.
- Synthetic calibrated runs may use L2; photo-facing paths must support nuisance-invariant residuals.
- Physics-only inversion is reported separately from posterior inference.
- Results are latent/uncertainty/extraction metrics, not aesthetic image scores.
- All benchmark runs are seed-controlled and JSON-receipted.
- Negative results are preserved as the scientific outcome.

## Review Focus

1. **Measurement leakage:** hidden occupancy must never enter the visible crop or extracted boundary signal except through the declared transport.
2. **Nuisance cheating:** affine residual must actually remove global exposure/offset nuisance without erasing chromatic structure.
3. **No-occluder honesty:** removing angular visibility structure must destroy or sharply weaken positional information on asymmetric cases.
4. **Extraction confound:** synthetic pixel extraction must be scored against known `y_true`; inverse success from oracle `y` must not be mislabeled as success from pixels.
5. **Gradient/unit imbalance:** Gate 1 must compare normalized latent coordinates and verify autograd against finite differences.

---

### Task 1: Package scaffold and CLI contracts

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/sighextraimage/__init__.py`
- Create: `src/sighextraimage/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces CLI entry point `sighextraimage = sighextraimage.cli:main`.
- Produces subcommands `synth`, `extract`, `invert`, `gate0`, `gate1`, `benchmark`.

- [ ] **Step 1:** Write failing parser tests for `--help` and all subcommands.
- [ ] **Step 2:** Run `pytest tests/test_cli.py -q`; expect import/parser failure.
- [ ] **Step 3:** Implement package metadata, `build_parser() -> argparse.ArgumentParser`, and `main(argv: list[str] | None = None) -> int`; command bodies may be stubs until owning tasks land.
- [ ] **Step 4:** Run `pytest tests/test_cli.py -q`; expect PASS.
- [ ] **Step 5:** Commit as `chore: scaffold SighExtraImage core`.

### Task 2: Physical latent, scene rendering, and corner transport

**Files:**
- Create: `src/sighextraimage/latent.py`
- Create: `src/sighextraimage/scene.py`
- Create: `src/sighextraimage/transport.py`
- Test: `tests/test_latent.py`
- Test: `tests/test_scene.py`
- Test: `tests/test_transport.py`

**Interfaces:**
- Produces `PhysicalLatent(theta, width, height, rgb, brightness, shape_code)` with batch tensors.
- Produces `SceneConfig` and `TransportConfig`.
- Produces `sample_prior(n, generator, device='cpu') -> PhysicalLatent`.
- Produces `render_hidden(latent, scene, transport) -> Tensor[batch,3,H,W_hidden]`.
- Produces `render_visible_with_leakage(latent, scene, transport, noise_generator) -> SyntheticObservation` containing full truth, visible crop, `R`, `y_true`, and noisy `y`.
- Produces `CornerTransport(config)` and `NoOccluderTransport(config)` with `forward_hidden(hidden_img) -> Tensor[M,3]` and `forward_latent(latent, scene) -> Tensor[batch,M,3]`.
- Produces `condition_spectrum() -> Tensor` / `condition_number() -> float`.

- [ ] **Step 1:** Write failing latent/scene tests for deterministic sampling, valid ranges, shapes, and zero hidden overlap with visible crop.
- [ ] **Step 2:** Run `pytest tests/test_latent.py tests/test_scene.py -q`; expect FAIL.
- [ ] **Step 3:** Implement latent dataclass/ranges and differentiable hidden renderer with disk/soft-rectangle occupancy.
- [ ] **Step 4:** Run latent/scene tests; expect PASS.
- [ ] **Step 5:** Write failing transport tests: zero brightness gives zero object leakage; RGB channel changes propagate; angular position moves boundary profile; correct operator deterministic; no-occluder singular spectrum is more rank-poor / position-insensitive on declared case.
- [ ] **Step 6:** Run `pytest tests/test_transport.py -q`; expect FAIL.
- [ ] **Step 7:** Implement smoothed corner visibility matrix `K`, distance falloff, ambient/gain, and no-occluder all-visible visibility control.
- [ ] **Step 8:** Run transport tests; expect PASS.
- [ ] **Step 9:** Commit as `feat: add physical latent and corner transport`.

### Task 3: Boundary extraction and nuisance-invariant likelihoods

**Files:**
- Create: `src/sighextraimage/extraction.py`
- Create: `src/sighextraimage/likelihood.py`
- Test: `tests/test_extraction.py`
- Test: `tests/test_likelihood.py`

**Interfaces:**
- Produces `extract_boundary_signal(image: Tensor, region: BoundaryRegion, *, n_measure: int, smooth_sigma: float, mode: str='log_lowpass') -> Tensor[M,3]`.
- Produces reserved `extract_boundary_signal_temporal(frames, region, ...) -> Tensor[M,3]`.
- Produces `physics_residual(y, yhat, mode: Literal['l2','affine','affine_per_channel']) -> Tensor`.
- Produces extraction metrics `profile_correlation`, `profile_rmse`.

- [ ] **Step 1:** Write failing extraction test on a synthetic rendered crop with texture nuisance; extracted profile must correlate positively with known `y_true` and respond to changed hidden color/position.
- [ ] **Step 2:** Run `pytest tests/test_extraction.py -q`; expect FAIL.
- [ ] **Step 3:** Implement sRGB-to-linear helper, strip aggregation, log-domain low-pass/detrending, and temporal extractor skeleton with real frame differences.
- [ ] **Step 4:** Run extraction tests; expect PASS.
- [ ] **Step 5:** Write failing likelihood tests: L2 zero at identity; affine residual invariant to shared gain + per-channel offset; affine retains penalty for changed profile shape; affine-per-channel is more color-blind than affine.
- [ ] **Step 6:** Run `pytest tests/test_likelihood.py -q`; expect FAIL.
- [ ] **Step 7:** Implement closed-form nuisance fits with stable epsilons and no gradient detaches.
- [ ] **Step 8:** Run likelihood tests; expect PASS.
- [ ] **Step 9:** Commit as `feat: add boundary extraction and invariant likelihoods`.

### Task 4: Physics-only TV inversion baseline

**Files:**
- Create: `src/sighextraimage/inversion.py`
- Test: `tests/test_inversion.py`

**Interfaces:**
- Produces `tv_invert(transport, y, *, lambda_tv: float, iters: int, lr: float, residual_mode: str) -> Tensor[J,3]`.
- Produces `profile_centroid(profile, theta) -> float` and `profile_excess_color(profile) -> Tensor[3]`.

- [ ] **Step 1:** Write failing inversion tests: low-noise corner case recovers coarse angular centroid better than a flat profile; no-occluder cannot localize the same case comparably; result stays non-negative/finite.
- [ ] **Step 2:** Run `pytest tests/test_inversion.py -q`; expect FAIL.
- [ ] **Step 3:** Implement nonnegative radiance optimization with TV penalty and declared residual mode.
- [ ] **Step 4:** Run inversion tests; expect PASS.
- [ ] **Step 5:** Commit as `feat: add physics-only inversion baseline`.

### Task 5: Gate 0 posterior inference and controls

**Files:**
- Create: `src/sighextraimage/prior.py`
- Create: `src/sighextraimage/inference.py`
- Test: `tests/test_inference.py`

**Interfaces:**
- Produces `ToyScenePrior.sample(n, generator) -> PhysicalLatent` and `render(latent) -> hidden image`.
- Produces `normalize_log_weights(log_w) -> weights`.
- Produces `PosteriorSummary` with mean/median latent, shape probability, entropy, ESS, credible widths.
- Produces `evaluate_gate0(z_true, y, candidates, transports, scene, residual_mode, sigma) -> Gate0Result` for prior/correct/wrong/no-occluder on the exact same candidates.

- [ ] **Step 1:** Write failing numerical tests for finite normalized weights, additive-constant invariance, and zero-information posterior = prior.
- [ ] **Step 2:** Run targeted tests; expect FAIL.
- [ ] **Step 3:** Implement weighted summaries, entropy, ESS, and stable log-weight normalization.
- [ ] **Step 4:** Run targeted tests; expect PASS.
- [ ] **Step 5:** Write failing Gate 0 tests: exact candidate identity across arms; low-noise correct physics contracts theta/color more than prior; correct beats wrong/no-occluder on declared asymmetric case.
- [ ] **Step 6:** Run `pytest tests/test_inference.py -q`; expect FAIL.
- [ ] **Step 7:** Implement `evaluate_gate0` without resampling before comparison.
- [ ] **Step 8:** Run inference tests; expect PASS.
- [ ] **Step 9:** Commit as `feat: add Gate 0 posterior controls`.

### Task 6: Gate 1 gradient-direction falsifier

**Files:**
- Create: `src/sighextraimage/gradient_gate.py`
- Test: `tests/test_gradient_gate.py`

**Interfaces:**
- Produces normalized continuous latent vector `[theta,width,height,r,g,b,brightness]`; shape held fixed.
- Produces `boundary_loss(latent, y, transport, scene, residual_mode) -> Tensor`.
- Produces `evaluate_gate1(...) -> Gate1Result` with correct, wrong, no-occluder, and norm-matched random controls.

- [ ] **Step 1:** Write failing autograd-vs-finite-difference tests for theta, width, RGB, brightness on an asymmetric case.
- [ ] **Step 2:** Run `pytest tests/test_gradient_gate.py -q`; expect FAIL.
- [ ] **Step 3:** Implement packing/unpacking and differentiable loss with prior-range normalization.
- [ ] **Step 4:** Run derivative tests; expect PASS.
- [ ] **Step 5:** Add failing behavior tests: one small correct step reduces normalized truth error for at least the declared identifiable coordinates; wrong/no-occluder/random controls perform worse on that case.
- [ ] **Step 6:** Run gradient suite; expect FAIL for missing evaluator.
- [ ] **Step 7:** Implement one-step and short-trajectory evaluator with componentwise diagnostics.
- [ ] **Step 8:** Run gradient suite; expect PASS.
- [ ] **Step 9:** Commit as `feat: add Gate 1 gradient falsifier`.

### Task 7: Metrics, receipts, benchmark, and extraction ablations

**Files:**
- Create: `src/sighextraimage/metrics.py`
- Create: `src/sighextraimage/receipts.py`
- Create: `src/sighextraimage/benchmark.py`
- Modify: `src/sighextraimage/cli.py`
- Test: `tests/test_metrics.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Produces `run_benchmark(seeds, candidates, sigma, residual_mode, output) -> dict`.
- Receipt contains scene/transport configs, extraction-vs-oracle metrics, TV baseline, Gate 0 arms, Gate 1 controls, condition spectrum summaries, instability count.
- Benchmark exposes ablations: ideal oracle `y`, extracted synthetic `y`, added noise, JPEG/quantization nuisance, no occluder.

- [ ] **Step 1:** Write failing metric tests for truth-zero errors, contraction ratio, extraction correlation, and condition summaries.
- [ ] **Step 2:** Run metrics tests; expect FAIL.
- [ ] **Step 3:** Implement metrics and JSON-safe serialization.
- [ ] **Step 4:** Run metrics tests; expect PASS.
- [ ] **Step 5:** Write failing benchmark reproducibility tests on 3 tiny seeds, including all four Gate 0 arms and oracle-vs-extracted measurement labels.
- [ ] **Step 6:** Run benchmark tests; expect FAIL.
- [ ] **Step 7:** Implement benchmark orchestration and wire CLI commands.
- [ ] **Step 8:** Run benchmark tests and full `pytest -q`; expect PASS.
- [ ] **Step 9:** Commit as `feat: add deterministic scientific benchmark`.

### Task 8: First declared scientific run

**Files:**
- Create: `results/receipts/gates-0-1-v0.json`
- Modify: `README.md`

- [ ] **Step 1:** Freeze seed set, candidate count, noise, transport geometry, extraction settings, residual mode, TV hyperparameters, Gate 1 offsets/step size in README before the run.
- [ ] **Step 2:** Run `sighextraimage benchmark ... --output results/receipts/gates-0-1-v0.json`.
- [ ] **Step 3:** Reject interpretation if non-finite values, hidden leakage, or implementation faults appear; debug first.
- [ ] **Step 4:** Report separately: oracle-y Gate 0, extracted-y Gate 0, TV baseline, no-occluder ablation, Gate 1 correct/wrong/no-occluder/random.
- [ ] **Step 5:** Update README with exact receipt values and narrow claim, including null attributes and extraction failures.
- [ ] **Step 6:** Run `pytest -q` again; expect PASS.
- [ ] **Step 7:** Commit as `results: record SighExtraImage Gates 0-1 v0`.

## Self-review outcome

- Revised spec coverage for the scientific core is complete: structured/no-occluder transport, extraction, nuisance-invariant likelihood, TV inversion, Gate 0, Gate 1, conditioning diagnostics, deterministic receipts, and negative-result preservation all have owning tasks.
- GUI is intentionally moved to the companion GUI plan and depends only on these public interfaces.
- Diffusion remains excluded until Gates 0–2 produce an interpretable receipt.
- Type/function names are consistent across tasks and every Review Focus item has an owning test.
