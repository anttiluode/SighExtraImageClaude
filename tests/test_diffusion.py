import json
import numpy as np
import pytest
import torch

from sighextraimage import diffusion as diffusion
from sighextraimage.outpainting import DEFAULT_MODEL, CoarseLightConstraint, OutpaintConfig, prepare_canvas
from sighextraimage.extraction import srgb_to_linear
from sighextraimage.transport import CornerTransport, TransportConfig


@pytest.fixture(params=[4,9])
def tiny_pipeline(tmp_path, request):
    pytest.importorskip('diffusers')
    from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionInpaintPipeline, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
    from transformers.models.clip.tokenization_clip import bytes_to_unicode
    torch.manual_seed(8)
    tokens = list(bytes_to_unicode().values())
    vocab = {t:i for i,t in enumerate(tokens + [t+'</w>' for t in tokens] + ['<|startoftext|>','<|endoftext|>'])}
    (tmp_path/'vocab.json').write_text(json.dumps(vocab))
    (tmp_path/'merges.txt').write_text('#version: 0.2\n')
    tokenizer = CLIPTokenizer(str(tmp_path/'vocab.json'),str(tmp_path/'merges.txt'),model_max_length=77)
    text_encoder = CLIPTextModel(CLIPTextConfig(vocab_size=len(vocab),hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=4,max_position_embeddings=77,bos_token_id=512,eos_token_id=513,pad_token_id=513))
    vae = AutoencoderKL(block_out_channels=(8,16,16,16),layers_per_block=1,latent_channels=4,
        down_block_types=('DownEncoderBlock2D',)*4,up_block_types=('UpDecoderBlock2D',)*4,norm_num_groups=4,sample_size=64)
    unet = UNet2DConditionModel(sample_size=64,in_channels=request.param,out_channels=4,layers_per_block=1,
        block_out_channels=(16,32),down_block_types=('DownBlock2D','CrossAttnDownBlock2D'),
        up_block_types=('CrossAttnUpBlock2D','UpBlock2D'),cross_attention_dim=16,attention_head_dim=4,norm_num_groups=4)
    scheduler = DDIMScheduler(steps_offset=1,clip_sample=False)
    return StableDiffusionInpaintPipeline(vae=vae,unet=unet,text_encoder=text_encoder,tokenizer=tokenizer,
        scheduler=scheduler,safety_checker=None,feature_extractor=None,requires_safety_checker=False)


def test_text_guidance_moves_toward_conditional_prediction():
    actual = diffusion.classifier_free_prediction(torch.tensor([1.]),torch.tensor([3.]),7)
    torch.testing.assert_close(actual, torch.tensor([15.]))


def test_guidance_step_is_masked_and_energy_bounded():
    gradient = torch.arange(64,dtype=torch.float32).reshape(1,4,4,4)
    mask = torch.zeros(1,1,4,4)
    mask[:,:,:,2:] = 1
    step = diffusion.bounded_light_step(gradient,mask,noise_scale=0.4,strength=0.08)
    assert not step[:,:,:,:2].any()
    assert step.norm() <= 0.08*0.4*(32**0.5)+1e-7
    assert step.norm() > 0
    assert not diffusion.bounded_light_step(gradient*float('nan'),mask,noise_scale=1,strength=0.08).any()


def test_pipeline_prompt_embeddings_keep_unconditional_first(tiny_pipeline,monkeypatch):
    positive = torch.full((1,77,16),3.)
    negative = torch.full((1,77,16),1.)
    monkeypatch.setattr(tiny_pipeline,'encode_prompt',lambda *args,**kwargs:(positive,negative))
    engine = diffusion.DiffusionEngine(tiny_pipeline)
    emb = engine._prompt_embeddings('room')
    assert emb[0].mean() == 1
    assert emb[1].mean() == 3


def test_real_diffusers_loop_is_seeded_and_preserves_visible_pixels(tiny_pipeline):
    prepared = prepare_canvas(np.full((40,80,3),128,dtype=np.uint8),OutpaintConfig(extension_fraction=0.25,max_side=128))
    engine = diffusion.DiffusionEngine(tiny_pipeline)
    config = OutpaintConfig(steps=2,max_side=128,seed=4)
    first = engine.generate(prepared,config)
    second = engine.generate(prepared,config)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    assert first.shape == prepared.canvas.shape
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first[:,:40,:80],prepared.canvas[:,:40,:80],rtol=0,atol=0)
    third = engine.generate(prepared,OutpaintConfig(steps=2,max_side=128,seed=5))
    assert not torch.equal(first[:,:,80:],third[:,:,80:])


def test_real_diffusers_guidance_is_finite_and_does_not_train_weights(tiny_pipeline):
    prepared = prepare_canvas(np.full((40,80,3),128,dtype=np.uint8),OutpaintConfig(max_side=128))
    transport = CornerTransport(TransportConfig(n_measure=32,n_angle=32))
    hidden = torch.linspace(0.1,0.9,48).expand(3,24,48).clone()
    y = transport.forward_hidden(srgb_to_linear(hidden))[0]
    constraint = CoarseLightConstraint.from_measurement(y,transport)
    engine = diffusion.DiffusionEngine(tiny_pipeline)
    result = engine.generate(prepared,OutpaintConfig(steps=4,max_side=128),constraint=constraint)
    assert torch.isfinite(result).all()
    assert engine.last_run['guidance_steps'] > 0
    assert all(p.grad is None and not p.requires_grad for p in tiny_pipeline.unet.parameters())


def test_real_safetensors_checkpoint_runs_through_headless_command(tiny_pipeline,tmp_path):
    from PIL import Image
    from sighextraimage.cli import main
    model = tmp_path/'tiny-model'
    tiny_pipeline.save_pretrained(model,safe_serialization=True)
    source = tmp_path/'source.png'
    output = tmp_path/'extension.png'
    guided = tmp_path/'comparison.png'
    image = np.full((40,80,3),128,dtype=np.uint8)
    Image.fromarray(image).save(source)
    assert main(['outpaint',str(source),'--output',str(output),'--guided-output',str(guided),
                 '--model',str(model),'--steps','2','--max-side','128']) == 0
    result = np.asarray(Image.open(output))
    assert result.shape == (40,116,3)
    np.testing.assert_array_equal(result[:,:80],image)
    assert not np.all(result[:,80:] == 128)
    np.testing.assert_array_equal(result,np.asarray(Image.open(guided)))


def test_sample_prediction_preserves_generation_and_explicitly_skips_guidance(tiny_pipeline):
    tiny_pipeline.scheduler.register_to_config(prediction_type='sample')
    engine = diffusion.DiffusionEngine(tiny_pipeline)
    prepared = prepare_canvas(np.full((40,80,3),128,dtype=np.uint8),OutpaintConfig(max_side=128))
    transport = CornerTransport(TransportConfig(n_measure=32,n_angle=32))
    hidden = torch.linspace(0.1,0.9,48).expand(3,24,48).clone()
    y = transport.forward_hidden(srgb_to_linear(hidden))[0]
    constraint = CoarseLightConstraint.from_measurement(y,transport)
    result = engine.generate(prepared,OutpaintConfig(steps=4,max_side=128),constraint=constraint)
    assert torch.isfinite(result).all()
    assert engine.last_run['guidance_steps'] == 0
    assert 'sample' in engine.last_run['guidance_disabled_reason'].lower()


def test_default_model_loads_fp16_variant_and_generates_on_cpu(tiny_pipeline,tmp_path,monkeypatch):
    from diffusers import StableDiffusionInpaintPipeline
    model = tmp_path/'fp16-only-model'
    for component in (tiny_pipeline.unet,tiny_pipeline.vae,tiny_pipeline.text_encoder):
        component.half()
    tiny_pipeline.save_pretrained(model,safe_serialization=True,variant='fp16')
    load_checkpoint = StableDiffusionInpaintPipeline.from_pretrained

    # Replace only the remote model location; keep real checkpoint loading,
    # dtype conversion, scheduler setup, and generation.
    def local_default_checkpoint(cls,model_id,**kwargs):
        assert model_id == DEFAULT_MODEL
        return load_checkpoint(model,**kwargs)
    monkeypatch.setattr(StableDiffusionInpaintPipeline,'from_pretrained',classmethod(local_default_checkpoint))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:False)

    engine = diffusion.DiffusionEngine.from_model(DEFAULT_MODEL)
    assert engine.device.type == 'cpu'
    assert engine.unet_dtype == torch.float32
    assert next(engine.pipe.text_encoder.parameters()).dtype == torch.float32
    config = OutpaintConfig(steps=2,max_side=128)
    prepared = prepare_canvas(np.full((40,80,3),128,dtype=np.uint8),config)
    result = engine.generate(prepared,config)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result[:,:40,:80],prepared.canvas[:,:40,:80],rtol=0,atol=0)
