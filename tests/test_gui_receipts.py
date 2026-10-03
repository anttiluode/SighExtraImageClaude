import json
import numpy as np

from sighextraimage.gui import gui_diagnostic_receipt


def test_gui_receipt_is_json_safe_and_does_not_embed_image_bytes():
    diag = {
        'signal_strength': 0.2,
        'profile': np.zeros((8,3), dtype=np.float32),
        'warnings': ['weak'],
        'raw_image': np.zeros((20,20,3), dtype=np.uint8),
    }
    receipt = gui_diagnostic_receipt('photo_gui', diag)
    text = json.dumps(receipt)
    assert receipt['source_mode'] == 'photo_gui'
    assert 'raw_image' not in receipt['diagnostics']
    assert 'profile' in receipt['diagnostics']
    assert len(text) < 10000
