# Photo outpainting

The user wants SighExtraImage to produce an actual extension outside a photograph's field of view. Varjoluotain is the inspiration and is read-only for this task. The previously discussed comparison (prior-only extension, optional light-guided extension, evidence diagnostics) was approved by the user's instruction to implement it.

## Behavior

- Photo Inspector gains a Generate extension action, prompt, edge, extension fraction, seed, steps, and an optional experimental light-guidance toggle.
- Prior-only generation always runs when the image and model are valid. Weak or flat boundary evidence never blocks ordinary generation.
- The output is larger on the chosen edge; every original photo pixel is preserved exactly. Working images retain aspect ratio.
- The generator downloads and caches the public Stable Diffusion 1.5 inpainting checkpoint on first use; the default ID is `stable-diffusion-v1-5/stable-diffusion-inpainting`. CPU, CUDA, and MPS are supported.
- Optional guidance uses SighExtraImage's existing extraction and CornerTransport. It discards weak singular modes, respects positive exposure scaling, gates gradients to the generated region, and bounds update energy. This is an experimental constraint under assumed geometry, not a calibrated hidden-scene reconstruction.
- Flat/nonfinite measurements skip guidance and return the prior image in the comparison panel with an explicit reason. Guidance errors preserve an already generated prior output.
- Identical seeds and deterministic VAE encodings make the prior/guided comparison reproducible.
- A headless `outpaint` command supports saved outputs. Installation uses an optional `outpaint` dependency extra; the existing scientific core and first receipt stay unchanged.

## Architecture

`outpainting.py` handles canvas geometry, coarse measurement loss, and comparison orchestration. `diffusion.py` is the lazy optional Diffusers adapter and owns the denoising loop/cache. `gui.py` and `cli.py` consume those public interfaces. Models are frozen, and autograd is enabled only for the bounded light update.

## Validation

Tests cover all four extension edges and exact pixel preservation, aspect ratio/padding, weak-evidence fallback, correct CFG ordering, deterministic seeds, finite bounded masked guidance, prior/guided callback behavior, CLI output, and a real Diffusers pipeline constructed locally with tiny random weights. A downloaded model smoke run is attempted if network/model access is available; its absence is reported explicitly and is not substituted with a quality claim.
