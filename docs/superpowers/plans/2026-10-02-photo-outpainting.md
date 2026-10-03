# Photo Outpainting Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan inline, task by task.

**Goal:** Generate actual photo extensions in SighExtraImage with optional measurement guidance.

**Architecture:** A geometry/comparison module uses the existing extraction/transport; a lazy cached Diffusers adapter performs deterministic inpainting and bounded light updates. GUI and CLI share that workflow.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, Pillow, Gradio 5, Diffusers 0.35, Transformers 4.

**Spec:** `docs/superpowers/specs/2026-10-02-photo-outpainting-design.md`.

## Global Constraints

- Change SighExtraImage only; Varjoluotain is a reference.
- Generate prior-only output even when evidence is weak.
- Preserve original pixels, aspect ratio, and all four selected edges.
- Use the scientific core without changing its first results/receipt.
- Label light guidance experimental; never claim recovered ground truth.
- Models load lazily; test runs need no model download unless explicitly running the model smoke check.

## Review Focus

1. Non-square images, all four edges, and dimensions not divisible by eight.
2. Flat/noisy/nonfinite signals and weak transport modes must not create confident guidance.
3. Correct positive/unconditional embedding order and seeded noise.
4. Missing model dependencies/download failures and a failed guided run must have useful user-visible results.
5. GPU/MPS device/dtype conversion and concurrent requests sharing a scheduler.

### Task 1: Canvas and measurement contract

Files: create `src/sighextraimage/outpainting.py`, `tests/test_outpainting.py`.

Interfaces: `OutpaintConfig`; `prepare_canvas(image, config) -> OutpaintCanvas`; `restore_outpaint(generated, prepared) -> np.ndarray`; `CoarseLightConstraint` computes a differentiable normalized shape residual using retained singular modes.

- [x] Write/run failing tests for each edge's enlarged dimensions and exact original pixels, aspect ratio, invalid controls, flat fallback, discarded singular modes, and finite positive-scale loss.
- [x] Implement the canvas and coarse constraint; run focused tests.
- [x] Commit verified behavior.

### Task 2: Real generator and comparison

Files: create `src/sighextraimage/diffusion.py`, `tests/test_diffusion.py`; extend `outpainting.py` and its tests.

Interfaces: `DiffusionEngine(pipe)` accepts a real pipeline; `get_engine(model_id)` lazily loads/caches it; `engine.generate(prepared, config, constraint=None, progress=None) -> torch.Tensor`; `generate_outpaintings(image, photo_config, config, engine=None, progress=None) -> OutpaintComparison`.

- [x] Write/run failing tests for CFG polarity, deterministic sampling, bounded masked updates, prior-only generation on flat input, guidance application/skip, and preserving prior output after a guided failure.
- [x] Implement frozen-model DDIM denoising for four-/nine-channel UNets, deterministic VAE modes, original-pixel composition, and cached serialized execution.
- [x] Exercise the loop with real tiny Diffusers UNet/VAE/scheduler/text components; run focused tests and commit.

### Task 3: User entry points and delivery

Files: modify `gui.py`, `cli.py`, `pyproject.toml`, `README.md`; create `tests/test_gui_outpainting.py`, `tests/test_cli_outpainting.py`.

- [x] Write/run failing GUI/CLI behavior tests for invoking generation and returning/saving an enlarged image.
- [x] Wire shared controls and comparison outputs; add optional dependencies and first-use download/cache instructions.
- [x] Run the complete suite, compile check, and GUI launch/client smoke. Optional full-checkpoint smoke was not run.
- [x] Review the full diff, sync through GitHub, and verify the remote tree includes the generator.

Verification: 80 tests pass, including real four-/nine-channel Diffusers components and a saved safetensors checkpoint through the CLI. A live Gradio client generated a 100 x 40 extension from an 80 x 40 photograph, preserved all original pixels exactly, and reported the weak-evidence fallback. Compile and whitespace checks pass. Full pretrained image quality and CUDA/MPS execution remain untested in this CPU environment.

One independent final review found no critical or important issues. Its two minor findings were addressed in one regression-tested pass: run diagnostics are captured under the generation lock, and clean-sample prediction checkpoints explicitly retain prior-only generation when the approximate light update is unavailable.
