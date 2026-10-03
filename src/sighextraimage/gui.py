from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import gradio as gr

from .extraction import extract_boundary_signal
from .gradient_gate import evaluate_gate1
from .inference import evaluate_gate0
from .inversion import tv_invert
from .latent import PhysicalLatent, sample_prior
from .scene import SceneConfig, render_visible_with_leakage
from .transport import CornerTransport, NoOccluderTransport, WrongCornerTransport, TransportConfig
from .photo import PhotoConfig, inspect_photo, WEAK_EVIDENCE_WARNING
from .outpainting import DEFAULT_MODEL, OutpaintConfig, generate_outpaintings


@dataclass
class SyntheticLabParams:
    theta: float = 0.82
    width: float = 0.11
    height: float = 0.50
    rgb: tuple[float, float, float] = (0.82, 0.16, 0.10)
    brightness: float = 0.95
    shape_code: int = 0
    noise: float = 0.0
    candidates: int = 600
    sigma: float = 0.01
    residual_mode: str = "l2"


@dataclass
class SyntheticLabView:
    full_truth: np.ndarray
    visible_crop: np.ndarray
    true_profile: np.ndarray
    extracted_profile: np.ndarray
    tv_profile: np.ndarray
    summary: dict
    synthetic_only: bool = True


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _image_hwc(x):
    a = _to_numpy(x)
    if a.ndim == 4:
        a = a[0]
    if a.ndim == 3 and a.shape[0] in (1, 3, 4):
        a = np.transpose(a, (1, 2, 0))
    return np.clip(a, 0.0, 1.0)


def run_synthetic_lab(params: SyntheticLabParams) -> SyntheticLabView:
    scene = SceneConfig(height=72, visible_width=96, hidden_width=64)
    cfg = TransportConfig(n_measure=64, n_angle=64, ambient=0.0, gain=1.0)
    correct = CornerTransport(cfg)
    wrong = WrongCornerTransport(cfg)
    no_occ = NoOccluderTransport(cfg)
    truth = PhysicalLatent.from_scalars(
        theta=params.theta,
        width=params.width,
        height=params.height,
        rgb=params.rgb,
        brightness=params.brightness,
        shape_code=params.shape_code,
    )
    obs = render_visible_with_leakage(truth, scene, correct, noise_std=params.noise, texture_seed=17)
    extracted = extract_boundary_signal(obs.visible_srgb, obs.region, n_measure=cfg.n_measure, smooth_sigma=1.5)
    y_true = obs.y_true[0] if hasattr(obs.y_true, "shape") and len(obs.y_true.shape) == 3 else obs.y_true
    tv = tv_invert(correct, torch.as_tensor(y_true), lambda_tv=0.015, iters=120, lr=0.08, residual_mode="l2")
    candidates = sample_prior(params.candidates, generator=torch.Generator().manual_seed(1234))
    gate0 = evaluate_gate0(truth, torch.as_tensor(y_true), candidates, correct, wrong, no_occ, scene, residual_mode=params.residual_mode, sigma=params.sigma)
    init_theta = max(0.12, params.theta - 0.28) if params.theta > 0.65 else min(1.18, params.theta + 0.28)
    init = PhysicalLatent.from_scalars(theta=init_theta, width=params.width, height=params.height, rgb=params.rgb, brightness=params.brightness, shape_code=params.shape_code)
    gate1 = evaluate_gate1(truth, init, torch.as_tensor(y_true), correct, wrong, no_occ, scene, residual_mode="l2", step_size=0.12, random_generator=torch.Generator().manual_seed(99))
    summary = {
        "posterior_contraction": float(gate0.correct.std_theta / max(gate0.prior.std_theta, 1e-12)),
        "wrong_contraction": float(gate0.wrong.std_theta / max(gate0.prior.std_theta, 1e-12)),
        "no_occluder_contraction": float(gate0.no_occluder.std_theta / max(gate0.prior.std_theta, 1e-12)),
        "gate1_correct_reduction": float(gate1.correct.error_reduction),
        "gate1_wrong_reduction": float(gate1.wrong.error_reduction),
        "gate1_no_occluder_reduction": float(gate1.no_occluder.error_reduction),
        "note": "Synthetic truth is evaluation-only and is unavailable for real photos.",
    }
    return SyntheticLabView(
        full_truth=_image_hwc(obs.full_truth_srgb),
        visible_crop=_image_hwc(obs.visible_srgb),
        true_profile=_to_numpy(y_true),
        extracted_profile=_to_numpy(extracted),
        tv_profile=_to_numpy(tv),
        summary=summary,
    )


def _synthetic_callback(theta, width, r, g, b, candidates):
    view = run_synthetic_lab(SyntheticLabParams(theta=theta, width=width, rgb=(r,g,b), candidates=int(candidates)))
    return view.full_truth, view.visible_crop, view.summary


def _photo_callback(image, edge, x0, y0, x1, y1, smooth_sigma, likelihood_mode):
    if image is None:
        return None, {"warnings": ["Upload a JPG or PNG first."]}
    cfg = PhotoConfig(
        edge=edge,
        region_fraction=(float(x0), float(y0), float(x1), float(y1)),
        smooth_sigma=float(smooth_sigma),
    )
    result = inspect_photo(image, cfg, likelihood_mode)
    diagnostics = {
        "signal_strength": result.signal_strength,
        "condition_number": result.condition_number,
        "warnings": result.warnings,
        "evidence_note": result.evidence_note,
        "profile": result.profile.tolist(),
        "derivative_profile": result.derivative_profile.tolist(),
        "tv_profile": None if result.tv_profile is None else result.tv_profile.tolist(),
    }
    return result.overlay_hwc, diagnostics


def gui_diagnostic_receipt(source_mode: str, diagnostics: dict) -> dict:
    """Return a compact JSON-safe GUI diagnostic receipt without raw image bytes."""
    from .receipts import jsonable
    allowed = {
        k: v for k, v in diagnostics.items()
        if k not in {"raw_image", "image", "image_bytes", "uploaded_image"}
    }
    return {
        "schema": "sighextraimage.gui-diagnostic.v0",
        "source_mode": str(source_mode),
        "diagnostics": jsonable(allowed),
    }


def _outpaint_callback(image, edge, x0, y0, x1, y1, smooth_sigma, prompt,
                       extension_fraction, steps, seed, light_guidance, physics_strength,
                       max_side, model_id, progress=gr.Progress()):
    if image is None:
        return None, None, "Upload a photograph first.", {}
    try:
        config = OutpaintConfig(edge=edge, extension_fraction=float(extension_fraction),
            prompt=prompt, steps=int(steps), seed=int(seed), light_guidance=bool(light_guidance),
            physics_strength=float(physics_strength), max_side=int(max_side), model_id=model_id)
        photo = PhotoConfig(edge=edge, region_fraction=(float(x0),float(y0),float(x1),float(y1)),
                            smooth_sigma=float(smooth_sigma))
        result = generate_outpaintings(image, photo, config,
            progress=lambda fraction, description: progress(fraction, desc=description))
        report = (f"Generated an extension on the **{edge}** edge. Original photo pixels are preserved.\n\n"
                  f"**Light guidance:** {result.guidance_status.replace('_',' ')}. "
                  "New pixels are generated hypotheses about the unseen region.")
        if result.warnings:
            report += "\n\n" + "\n".join("- " + warning for warning in result.warnings)
        return result.prior, result.guided, report, result.diagnostics
    except Exception as exc:
        return None, None, (f"Image generation could not complete: {exc}\n\n"
            "Install generation dependencies with `pip install -e '.[outpaint]'`. "
            "First use needs internet access to download the model. "
            "For memory errors, reduce the working resolution."), {}

def build_app() -> gr.Blocks:
    with gr.Blocks(title="SighExtraImage") as app:
        gr.Markdown("# SighExtraImage\nExtend photographs and inspect the boundary light that may constrain them.")
        with gr.Tab("Synthetic Lab"):
            gr.Markdown("Ground truth below is **synthetic-only** evaluation information.")
            with gr.Row():
                theta = gr.Slider(0.08, 1.22, value=0.82, label="Hidden angle")
                width = gr.Slider(0.07, 0.24, value=0.11, label="Hidden width")
                candidates = gr.Slider(64, 2500, value=600, step=64, label="Candidate hypotheses")
            with gr.Row():
                r = gr.Slider(0.05, 0.95, value=0.82, label="R")
                g = gr.Slider(0.05, 0.95, value=0.16, label="G")
                b = gr.Slider(0.05, 0.95, value=0.10, label="B")
            run = gr.Button("Run synthetic lab")
            with gr.Row():
                truth_img = gr.Image(label="Synthetic full truth")
                visible_img = gr.Image(label="Camera-visible crop")
            summary = gr.JSON(label="Diagnostics")
            run.click(_synthetic_callback, [theta, width, r, g, b, candidates], [truth_img, visible_img, summary])
        with gr.Tab("Photo Inspector"):
            gr.Markdown("Inspect boundary evidence, then generate an extension beyond the selected edge.")
            photo = gr.Image(type="numpy", label="Upload JPG/PNG")
            with gr.Row():
                edge = gr.Dropdown(["right", "left", "top", "bottom"], value="right", label="Boundary edge")
                likelihood = gr.Dropdown(["affine", "affine_per_channel", "l2"], value="affine", label="Likelihood nuisance mode")
                smooth = gr.Slider(0.0, 4.0, value=1.5, step=0.1, label="Boundary smoothing")
            gr.Markdown("Region is specified as normalized fractions of the displayed photo: x0, y0, x1, y1.")
            with gr.Row():
                x0 = gr.Slider(0.0, 0.95, value=0.55, step=0.01, label="x0")
                y0 = gr.Slider(0.0, 0.95, value=0.60, step=0.01, label="y0")
                x1 = gr.Slider(0.05, 1.0, value=0.98, step=0.01, label="x1")
                y1 = gr.Slider(0.05, 1.0, value=0.92, step=0.01, label="y1")
            inspect = gr.Button("Inspect boundary evidence")
            overlay = gr.Image(label="Canonicalized photo + selected region")
            photo_diag = gr.JSON(label="Boundary evidence diagnostics")
            inspect.click(_photo_callback, [photo, edge, x0, y0, x1, y1, smooth, likelihood], [overlay, photo_diag])
            gr.Markdown("### Extend the photo\n"
                "The first view uses the image model alone. The comparison can try experimental boundary-light guidance. "
                "When guidance is weak, skipped, or off, it repeats the prior-only view.")
            prompt = gr.Textbox(value=OutpaintConfig().prompt, label="Description of the extension", lines=2)
            with gr.Row():
                extend = gr.Slider(0.1,1.0,value=0.45,step=0.05,label="Amount to add (fraction of selected side)")
                steps = gr.Slider(5,60,value=30,step=1,label="Sampling steps")
                seed = gr.Number(value=42,precision=0,label="Random seed")
            with gr.Row():
                light_guidance = gr.Checkbox(value=True,label="Try boundary-light guidance (experimental)")
                physics_strength = gr.Slider(0,0.25,value=0.08,step=0.01,label="Light guidance strength")
                max_side = gr.Dropdown([256,384,512,640,768],value=512,label="Working resolution (maximum side)")
            with gr.Accordion("Image model",open=False):
                model_id = gr.Textbox(value=DEFAULT_MODEL,label="Hugging Face model ID or local model folder")
                gr.Markdown("The model downloads on first generation and is cached for subsequent runs. CPU generation can be slow.")
            generate = gr.Button("Generate image extension",variant="primary")
            with gr.Row():
                prior_image = gr.Image(label="Prior-only extension",format="png")
                guided_image = gr.Image(label="Light-guided comparison",format="png")
            generation_report = gr.Markdown()
            generation_diagnostics = gr.JSON(label="Generation diagnostics")
            generate.click(_outpaint_callback,
                [photo,edge,x0,y0,x1,y1,smooth,prompt,extend,steps,seed,light_guidance,physics_strength,max_side,model_id],
                [prior_image,guided_image,generation_report,generation_diagnostics],
                api_name="generate_extension",concurrency_limit=1)
    return app


def launch_gui(*, share: bool = False, server_name: str = "127.0.0.1", server_port: int | None = None):
    return build_app().queue().launch(share=share, server_name=server_name, server_port=server_port)
