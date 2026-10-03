import numpy as np
import pytest

from sighextraimage.photo import PhotoConfig, prepare_photo


@pytest.mark.parametrize('edge', ['left','right','top','bottom'])
def test_prepare_photo_maps_all_edges_to_valid_canonical_region(edge):
    img = np.zeros((120, 80, 3), dtype=np.uint8)
    cfg = PhotoConfig(edge=edge, region_fraction=(0.55, 0.55, 0.98, 0.92), max_side=64)
    p = prepare_photo(img, cfg)
    assert p.image_chw.shape[0] == 3
    assert max(p.image_chw.shape[1:]) <= 64
    assert 0 <= p.region.x0 < p.region.x1 <= p.image_chw.shape[2]
    assert 0 <= p.region.y0 < p.region.y1 <= p.image_chw.shape[1]
    assert p.overlay_hwc.shape[:2] == tuple(p.image_chw.shape[1:])


def test_prepare_photo_rejects_invalid_region():
    img = np.zeros((30, 40, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        prepare_photo(img, PhotoConfig(region_fraction=(0.8,0.8,0.2,0.9)))


def test_resize_preserves_normalized_region_semantics():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    p = prepare_photo(img, PhotoConfig(edge='right', region_fraction=(0.5,0.5,1.0,1.0), max_side=100))
    h,w = p.image_chw.shape[1:]
    assert abs(p.region.x0 / w - 0.5) < 0.03
    assert abs(p.region.y0 / h - 0.5) < 0.03
