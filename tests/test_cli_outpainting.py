import numpy as np
from PIL import Image

from sighextraimage import cli, outpainting


def test_headless_outpaint_saves_an_extended_photo(tmp_path,monkeypatch):
    source = tmp_path/'source.png'
    output = tmp_path/'extended.png'
    Image.fromarray(np.full((40,80,3),128,dtype=np.uint8)).save(source)
    def generate(image,photo,config,**kwargs):
        assert config.edge == 'right'
        assert config.seed == 7
        assert not config.light_guidance
        arr = np.array(image)
        result = np.pad(arr,((0,0),(0,20),(0,0)),constant_values=100)
        return outpainting.OutpaintComparison(result,result.copy(),'not_requested',[],{})
    monkeypatch.setattr(outpainting,'generate_outpaintings',generate)
    assert cli.main(['outpaint',str(source),'--output',str(output),'--extend','0.25','--seed','7']) == 0
    actual = np.array(Image.open(output))
    assert actual.shape == (40,100,3)
    np.testing.assert_array_equal(actual[:,:80],np.array(Image.open(source)))
