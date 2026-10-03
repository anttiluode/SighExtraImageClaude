import types
import numpy as np

import sighextraimage.gui as gui


def test_synthetic_lab_callback_delegates_to_core(monkeypatch):
    calls=[]

    class FakeObs:
        full_truth_srgb = np.zeros((3,4,4), dtype=np.float32)
        visible_srgb = np.zeros((3,4,3), dtype=np.float32)
        region = types.SimpleNamespace(y0=1,y1=3,x0=1,x1=3)
        y_true = np.zeros((1,8,3), dtype=np.float32)

    monkeypatch.setattr(gui, 'render_visible_with_leakage', lambda *a, **k: (calls.append('render') or FakeObs()))
    monkeypatch.setattr(gui, 'extract_boundary_signal', lambda *a, **k: (calls.append('extract') or np.zeros((8,3),dtype=np.float32)))
    monkeypatch.setattr(gui, 'tv_invert', lambda *a, **k: (calls.append('invert') or np.zeros((8,3),dtype=np.float32)))
    monkeypatch.setattr(gui, 'evaluate_gate0', lambda *a, **k: (calls.append('gate0') or types.SimpleNamespace(prior=types.SimpleNamespace(std_theta=1.0), correct=types.SimpleNamespace(std_theta=0.5), wrong=types.SimpleNamespace(std_theta=1.2), no_occluder=types.SimpleNamespace(std_theta=1.0))))
    monkeypatch.setattr(gui, 'evaluate_gate1', lambda *a, **k: (calls.append('gate1') or types.SimpleNamespace(correct=types.SimpleNamespace(error_reduction=0.1), wrong=types.SimpleNamespace(error_reduction=-0.1), no_occluder=types.SimpleNamespace(error_reduction=0.0), random=types.SimpleNamespace(error_reduction=0.0))))

    view = gui.run_synthetic_lab(gui.SyntheticLabParams(candidates=32))
    assert calls == ['render','extract','invert','gate0','gate1']
    assert view.synthetic_only is True
    assert 'posterior_contraction' in view.summary
