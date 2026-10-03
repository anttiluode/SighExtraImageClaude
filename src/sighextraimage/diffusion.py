"""Lazy Stable Diffusion 1.5 inpainting with bounded experimental light updates."""
from __future__ import annotations

from functools import lru_cache
import math
from threading import RLock
import torch
import torch.nn.functional as F

from .outpainting import DEFAULT_MODEL, CoarseLightConstraint, OutpaintCanvas, OutpaintConfig


def classifier_free_prediction(unconditional: torch.Tensor, conditional: torch.Tensor,
                               scale: float) -> torch.Tensor:
    return unconditional + scale * (conditional - unconditional)


def bounded_light_step(gradient: torch.Tensor, mask: torch.Tensor, *, noise_scale: float,
                       strength: float) -> torch.Tensor:
    active = mask.expand_as(gradient)
    gated = gradient * active
    if not torch.isfinite(gated).all():
        return torch.zeros_like(gradient)
    norm = gated.norm()
    if float(norm) < 1e-12:
        return torch.zeros_like(gradient)
    # Bound RMS displacement over the generated latent cells relative to the
    # current noise scale. This is an energy bound, not a per-coordinate clamp.
    budget = strength * noise_scale * math.sqrt(float(active.sum()))
    return gated / norm * budget


class DiffusionEngine:
    def __init__(self, pipe):
        self.pipe = pipe
        self.device = next(pipe.unet.parameters()).device
        self.unet_dtype = next(pipe.unet.parameters()).dtype
        self.pipe.vae.to(dtype=torch.float32)
        for model in (pipe.unet, pipe.vae, pipe.text_encoder):
            if model is not None:
                model.eval().requires_grad_(False)
        self.scaling_factor = float(pipe.vae.config.scaling_factor)
        channels = pipe.unet.config.in_channels
        if channels not in (4, 9) or pipe.vae.config.latent_channels != 4:
            raise ValueError("Use a Stable Diffusion 1.x/2.x model with a four- or nine-channel UNet.")
        self.inpaint_unet = channels == 9
        self._lock = RLock()
        self.last_run: dict = {}

    @classmethod
    def from_model(cls, model_id: str) -> DiffusionEngine:
        try:
            from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
        except ImportError as exc:
            raise RuntimeError("Install image generation first: pip install -e '.[outpaint]'") from exc
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        # The default repository publishes safetensors only as the fp16 variant.
        # Select its filenames independently of CPU/MPS computation precision.
        variant = "fp16" if model_id == DEFAULT_MODEL else None
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id, torch_dtype=dtype, variant=variant, use_safetensors=True,
            safety_checker=None, requires_safety_checker=False,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.to(device)
        return cls(pipe)

    @torch.no_grad()
    def _prompt_embeddings(self, prompt: str) -> torch.Tensor:
        positive, negative = self.pipe.encode_prompt(
            prompt, device=self.device, num_images_per_prompt=1,
            do_classifier_free_guidance=True, negative_prompt="",
        )
        return torch.cat([negative, positive], dim=0)

    @torch.no_grad()
    def _encode(self, normalized_image: torch.Tensor) -> torch.Tensor:
        # Posterior mode avoids unseeded VAE draws in the paired comparison.
        return self.pipe.vae.encode(normalized_image.float()).latent_dist.mode() * self.scaling_factor

    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        return ((self.pipe.vae.decode(latent.float() / self.scaling_factor).sample + 1) / 2).clamp(0, 1)

    def generate(self, prepared: OutpaintCanvas, config: OutpaintConfig, *,
                 constraint: CoarseLightConstraint | None = None, progress=None) -> torch.Tensor:
        image, _ = self.generate_with_diagnostics(prepared, config, constraint=constraint, progress=progress)
        return image

    def generate_with_diagnostics(self, prepared: OutpaintCanvas, config: OutpaintConfig, *,
                                  constraint: CoarseLightConstraint | None = None, progress=None):
        with self._lock:
            image = self._generate(prepared, config, constraint=constraint, progress=progress)
            return image, dict(self.last_run)

    def _generate(self, prepared: OutpaintCanvas, config: OutpaintConfig, *, constraint, progress):
        scheduler = self.pipe.scheduler
        disabled_reason = None
        if constraint is not None and scheduler.config.prediction_type == "sample":
            disabled_reason = ("This checkpoint predicts clean samples directly; the approximate light-gradient "
                               "update is unavailable. Prior-only generation is retained.")
            constraint = None
        scheduler.set_timesteps(config.steps, device=self.device)
        timesteps = scheduler.timesteps
        embeddings = self._prompt_embeddings(config.prompt)
        image = prepared.canvas.to(self.device)[None]
        mask = prepared.mask.to(self.device)[None]
        normalized = image * 2 - 1
        known = self._encode(normalized)
        masked = self._encode(normalized * (1 - mask))
        mask_lat = F.interpolate(mask, size=known.shape[-2:], mode="nearest")
        light_mask = torch.zeros_like(mask)
        light_mask[:, :, :prepared.visible_height,
                   prepared.visible_width:prepared.visible_width + prepared.extension_width] = 1
        light_mask = F.interpolate(light_mask, size=known.shape[-2:], mode="nearest")
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        noise = torch.randn(known.shape, generator=generator, dtype=torch.float32).to(self.device)
        latent = noise * scheduler.init_noise_sigma
        latent = mask_lat * latent + (1 - mask_lat) * scheduler.add_noise(known, noise, timesteps[:1])
        guidance_steps = 0
        for i, timestep in enumerate(timesteps):
            fraction = i / max(len(timesteps) - 1, 1)
            guided = (constraint is not None and constraint.allowed and
                      config.physics_strength > 0 and 0.20 <= fraction < 0.95)
            latent = latent.detach().requires_grad_(guided)
            with torch.no_grad():
                model_input = scheduler.scale_model_input(torch.cat([latent, latent]), timestep)
                if self.inpaint_unet:
                    model_input = torch.cat([
                        model_input, torch.cat([mask_lat, mask_lat]), torch.cat([masked, masked]),
                    ], dim=1)
                eps = self.pipe.unet(model_input.to(self.unet_dtype), timestep,
                                     encoder_hidden_states=embeddings).sample.float()
                unconditional, conditional = eps.chunk(2)
                eps = classifier_free_prediction(unconditional, conditional, config.text_guidance)
            # DDIM supplies a differentiable x0 path for epsilon/v prediction.
            step = scheduler.step(eps, timestep, latent, eta=0.0)
            next_latent = step.prev_sample.detach()
            if guided:
                with torch.enable_grad():
                    clean = self._decode(step.pred_original_sample)[0]
                    hidden = clean[:, :prepared.visible_height,
                                   prepared.visible_width:prepared.visible_width + prepared.extension_width]
                    loss = constraint.loss(hidden)
                    if torch.isfinite(loss):
                        gradient = torch.autograd.grad(loss, latent)[0]
                        alpha = scheduler.alphas_cumprod[int(timestep)].to(latent)
                        delta = bounded_light_step(gradient, light_mask,
                            noise_scale=float((1 - alpha).sqrt()), strength=config.physics_strength)
                        if bool(delta.any()):
                            next_latent = next_latent - delta
                            guidance_steps += 1
            if i + 1 < len(timesteps):
                known_noisy = scheduler.add_noise(known, noise, timesteps[i + 1:i + 2])
                next_latent = mask_lat * next_latent + (1 - mask_lat) * known_noisy
            latent = next_latent.detach()
            if progress is not None:
                progress((i + 1) / len(timesteps), "Denoising image extension")
        with torch.no_grad():
            decoded = self._decode(latent)
            result = (mask * decoded + (1 - mask) * image)[0].float().cpu()
        self.last_run = {"guidance_steps": guidance_steps, "seed": config.seed, "device": str(self.device)}
        if disabled_reason is not None:
            self.last_run["guidance_disabled_reason"] = disabled_reason
        return result


@lru_cache(maxsize=1)
def _cached_engine(model_id: str) -> DiffusionEngine:
    return DiffusionEngine.from_model(model_id)


_LOAD_LOCK = RLock()


def get_engine(model_id: str) -> DiffusionEngine:
    with _LOAD_LOCK:
        return _cached_engine(model_id)
