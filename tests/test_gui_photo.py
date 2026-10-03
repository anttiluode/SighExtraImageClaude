import numpy as np

from sighextraimage.photo import PhotoConfig, inspect_photo, WEAK_EVIDENCE_WARNING


def test_flat_photo_emits_weak_evidence_warning_and_no_synthetic_truth():
    img = np.full((120,160,3), 128, dtype=np.uint8)
    result = inspect_photo(img, PhotoConfig(edge='right', region_fraction=(0.55,0.6,0.98,0.92), max_side=160), 'affine')
    assert WEAK_EVIDENCE_WARNING in result.warnings
    assert result.profile.shape[1] == 3
    assert result.overlay_hwc.shape == img.shape
    assert not hasattr(result, 'hidden_truth')
    assert not hasattr(result, 'full_truth')


def test_textured_gradient_photo_produces_diagnostics():
    h,w=120,160
    x=np.linspace(0,1,w)[None,:,None]
    base=np.ones((h,w,3),dtype=np.float32)*0.4
    base[:,:,0]+=0.25*x[:,:,0]
    base[:,:,1]+=0.08*x[:,:,0]
    img=(np.clip(base,0,1)*255).astype(np.uint8)
    result=inspect_photo(img, PhotoConfig(edge='right', region_fraction=(0.45,0.55,0.98,0.95), max_side=160), 'affine')
    assert result.signal_strength >= 0
    assert result.condition_number > 0
    assert len(result.derivative_profile) == len(result.profile)
    assert result.tv_profile is not None
