import numpy as np
import torch

from sighextraimage import gui
from sighextraimage.outpainting import OutpaintComparison


def test_generate_callback_returns_a_larger_photo_and_guidance_status(monkeypatch):
    image = np.zeros((40,80,3),dtype=np.uint8)
    def generate(img,photo,config,**kwargs):
        assert config.edge == 'left'
        assert photo.edge == 'left'
        assert config.prompt == 'a room'
        assert config.light_guidance
        extended = np.pad(img,((0,0),(20,0),(0,0)),constant_values=100)
        return OutpaintComparison(extended,extended.copy(),'skipped',['weak boundary'],{'seed':7})
    monkeypatch.setattr(gui,'generate_outpaintings',generate)
    prior,guided,report,diagnostics = gui._outpaint_callback(
        image,'left',0.55,0.6,0.98,0.92,1.5,'a room',0.25,30,7,True,0.08,512,
        'stable-diffusion-v1-5/stable-diffusion-inpainting')
    assert prior.shape == (40,100,3)
    np.testing.assert_array_equal(prior[:,20:],image)
    np.testing.assert_array_equal(prior,guided)
    assert 'skipped' in report.lower()
    assert 'weak boundary' in report
    assert diagnostics['seed'] == 7


def test_missing_photo_and_model_error_have_actionable_messages(monkeypatch):
    args = ('right',0.55,0.6,0.98,0.92,1.5,'room',0.45,30,42,False,0.08,512,'model')
    assert 'upload' in gui._outpaint_callback(None,*args)[2].lower()
    def fail(*args,**kwargs):
        raise RuntimeError('download unavailable')
    monkeypatch.setattr(gui,'generate_outpaintings',fail)
    result = gui._outpaint_callback(np.zeros((40,80,3),dtype=np.uint8),*args)
    assert result[0] is None
    assert 'download unavailable' in result[2]


def test_photo_ui_exposes_a_generation_endpoint_and_comparison_panels():
    config = gui.build_app().get_config_file()
    assert any(d.get('api_name') == 'generate_extension' for d in config['dependencies'])
    labels = [c.get('props',{}).get('label','') for c in config['components']]
    assert 'Prior-only extension' in labels
    assert 'Light-guided comparison' in labels

