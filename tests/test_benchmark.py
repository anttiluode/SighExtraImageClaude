import json
from pathlib import Path

from sighextraimage.benchmark import run_benchmark


def test_benchmark_writes_reproducible_receipt_with_all_controls(tmp_path):
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a = run_benchmark([3, 7], candidates=160, sigma=0.01, residual_mode="affine", output=a_path, tv_iters=60)
    b = run_benchmark([3, 7], candidates=160, sigma=0.01, residual_mode="affine", output=b_path, tv_iters=60)
    assert a_path.exists() and b_path.exists()
    assert a == b
    loaded = json.loads(a_path.read_text())
    assert loaded["schema"] == "sighextraimage.gates01.v0"
    assert loaded["seeds"] == [3, 7]
    assert len(loaded["scenes"]) == 2
    scene = loaded["scenes"][0]
    assert set(scene["gate0_oracle"]) == {"prior", "correct", "wrong", "no_occluder"}
    assert set(scene["gate1"]) == {"correct", "wrong", "no_occluder", "random"}
    assert "extraction" in scene and "tv_inversion" in scene and "conditioning" in scene
    assert scene["candidate_count"] == 160


def test_benchmark_cli_parser_accepts_declared_options():
    from sighextraimage.cli import build_parser
    args = build_parser().parse_args(["benchmark", "--seeds", "1", "2", "--candidates", "80", "--output", "x.json"])
    assert args.command == "benchmark"
    assert args.seeds == [1,2]
    assert args.candidates == 80
    assert args.output == "x.json"
