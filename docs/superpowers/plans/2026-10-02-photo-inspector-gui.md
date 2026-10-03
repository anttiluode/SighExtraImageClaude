# SighExtraImage Photo Inspector GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local Gradio application that exposes the verified SighExtraImage scientific core for synthetic exploration and real-photo boundary-signal inspection without creating a separate demo pipeline.

**Architecture:** `gui.py` is a thin adapter over the same scene, transport, extraction, inversion, likelihood, inference, gradient, metrics, and receipt functions used by CLI/tests. Synthetic Lab may display ground truth because it is known; Photo Inspector must clearly distinguish extracted evidence, model-dependent inversion, and unsupported hidden-world details.

**Tech Stack:** Python 3.11+, Gradio, PyTorch CPU, NumPy, Pillow; matplotlib optional for diagnostics.

**Spec:** `docs/superpowers/specs/2026-10-02-physics-guided-outpainting-design.md`

**Depends on:** `docs/superpowers/plans/2026-10-02-gates-0-1-physical-latent.md` Tasks 1–7 complete and public interfaces stable.

## Global Constraints

- GUI must call the scientific core; no duplicate physics/extraction/inference implementations.
- `sighextraimage gui` launches locally.
- Synthetic-only ground truth must be labeled as unavailable for real photos.
- Photo mode must not claim to reveal a unique hidden scene.
- Weak/ill-conditioned measurements must surface warnings rather than a confident picture.
- No pretrained diffusion dependency in this GUI version.

## Review Focus

1. **Image orientation/region mismatch:** user-selected edge and region must map correctly into extraction coordinates.
2. **Large uploads:** GUI must resize/limit inputs without silently changing the selected region semantics.
3. **Weak evidence:** signal-quality panel must return a warning rather than fabricated hidden content.
4. **Synthetic truth leakage into photo mode:** photo callbacks must never receive synthetic hidden truth fields.
5. **GUI/core drift:** smoke tests must monkeypatch core functions and prove callbacks delegate to them.

---

### Task 1: GUI application shell and CLI launch

**Files:**
- Create: `src/sighextraimage/gui.py`
- Modify: `src/sighextraimage/cli.py`
- Modify: `pyproject.toml` dependencies/extras
- Test: `tests/test_gui_smoke.py`

**Interfaces:**
- Produces `build_app() -> gr.Blocks`.
- Produces `launch_gui(*, share: bool=False, server_name: str='127.0.0.1', server_port: int|None=None)`.
- CLI subcommand `sighextraimage gui [--share] [--port N]`.

- [ ] **Step 1:** Write failing smoke tests that import `build_app`, verify two tabs named `Synthetic Lab` and `Photo Inspector`, and verify CLI `gui` parses without launching in parser tests.
- [ ] **Step 2:** Run `pytest tests/test_gui_smoke.py tests/test_cli.py -q`; expect FAIL.
- [ ] **Step 3:** Implement minimal Gradio Blocks shell and CLI launch wiring; no scientific callback logic yet.
- [ ] **Step 4:** Run smoke/parser tests; expect PASS.
- [ ] **Step 5:** Commit as `feat: scaffold SighExtraImage GUI`.

### Task 2: Synthetic Lab callbacks

**Files:**
- Modify: `src/sighextraimage/gui.py`
- Test: `tests/test_gui_synthetic.py`

**Interfaces:**
- Produces pure callback `run_synthetic_lab(params: SyntheticLabParams) -> SyntheticLabView`.
- `SyntheticLabView` contains display-ready arrays/tables for full truth, visible crop, region overlay, true/extracted profiles, transport spectrum, TV inversion, Gate 0 summaries, and Gate 1 diagnostics.

- [ ] **Step 1:** Write failing callback tests using tiny configs and monkeypatched public core calls; assert callback delegates to renderer, extractor, inversion, Gate 0, and Gate 1 exactly once and labels ground-truth outputs `synthetic_only`.
- [ ] **Step 2:** Run `pytest tests/test_gui_synthetic.py -q`; expect FAIL.
- [ ] **Step 3:** Implement parameter dataclass, callback orchestration, and display conversion helpers without duplicating algorithms.
- [ ] **Step 4:** Run callback tests; expect PASS.
- [ ] **Step 5:** Wire controls for angle/position, size, color, shape, noise, occluder toggle, penumbra softness, likelihood mode, and bounded candidate count.
- [ ] **Step 6:** Add GUI component assertions/smoke test for expected outputs and warnings.
- [ ] **Step 7:** Commit as `feat: add Synthetic Lab GUI`.

### Task 3: Photo region selection and extraction workflow

**Files:**
- Modify: `src/sighextraimage/gui.py`
- Create: `src/sighextraimage/photo.py`
- Test: `tests/test_photo.py`
- Test: `tests/test_gui_photo.py`

**Interfaces:**
- Produces `PhotoConfig(edge: Literal['left','right','top','bottom'], region_fraction: tuple[float,float,float,float], max_side: int=1024, smooth_sigma: float=...)`.
- Produces `prepare_photo(image, config) -> PreparedPhoto` preserving mapping from displayed/resized coordinates to extraction coordinates.
- Produces `inspect_photo(image, config, likelihood_mode) -> PhotoInspection` with overlay, extracted profile, derivative/chromatic diagnostics, signal/SNR proxy, conditioning, optional TV inverse, and warning list.

- [ ] **Step 1:** Write failing image-preparation tests for portrait/landscape inputs, all four edges, out-of-range region rejection/clamping policy, and resize coordinate preservation.
- [ ] **Step 2:** Run `pytest tests/test_photo.py -q`; expect FAIL.
- [ ] **Step 3:** Implement photo normalization/resizing and explicit region conversion.
- [ ] **Step 4:** Run photo tests; expect PASS.
- [ ] **Step 5:** Write failing inspection tests: output contains extracted profile and diagnostics; flat/uninformative image emits the exact warning `No evidence that this selected boundary strongly constrains the unseen region under the current model.`; no synthetic ground-truth field is present.
- [ ] **Step 6:** Run `pytest tests/test_gui_photo.py -q`; expect FAIL.
- [ ] **Step 7:** Implement `inspect_photo` by delegating to core extraction, transport diagnostics, likelihood helpers, and TV inversion; derive a conservative signal-quality score from declared diagnostics only.
- [ ] **Step 8:** Run photo GUI tests; expect PASS.
- [ ] **Step 9:** Wire upload, edge selector, region controls, smoothing, nuisance mode, and outputs in the Photo Inspector tab.
- [ ] **Step 10:** Commit as `feat: add real-photo boundary inspector`.

### Task 4: GUI receipts, examples, and end-to-end verification

**Files:**
- Modify: `src/sighextraimage/gui.py`
- Modify: `README.md`
- Test: `tests/test_gui_smoke.py`

**Interfaces:**
- GUI actions may export a JSON diagnostic receipt using the same receipt schema helpers as CLI, marked `source_mode: synthetic_gui` or `source_mode: photo_gui`.

- [ ] **Step 1:** Add failing test that a synthetic GUI run and photo inspection can serialize diagnostics without embedding raw uploaded image bytes.
- [ ] **Step 2:** Run GUI tests; expect FAIL.
- [ ] **Step 3:** Implement receipt export adapter and README launch instructions.
- [ ] **Step 4:** Run `pytest -q`; require PASS.
- [ ] **Step 5:** Launch `sighextraimage gui` locally and manually verify both tabs render, upload accepts a normal JPG/PNG, and weak-evidence warning is visible on a flat image.
- [ ] **Step 6:** Commit as `feat: complete SighExtraImage local GUI`.

## Self-review outcome

- GUI plan is deliberately downstream of the core plan and contains no duplicate scientific algorithms.
- Both requested modes are covered: a rich Synthetic Lab and an immediately usable real-photo inspector.
- Failure/weak-evidence behavior is explicitly tested.
- Constrained diffusion outpainting remains a later plan after the core gates produce interpretable results.
