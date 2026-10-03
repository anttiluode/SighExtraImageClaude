from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sighextraimage",
        description="SighExtraImage: physics-guided out-of-frame inference",
    )
    sub = parser.add_subparsers(dest="command")
    for name in ("synth", "extract", "invert", "gate0", "gate1"):
        sub.add_parser(name)
    gui = sub.add_parser("gui")
    gui.add_argument("--share", action="store_true")
    gui.add_argument("--port", type=int, default=None)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 11])
    bench.add_argument("--candidates", type=int, default=1200)
    bench.add_argument("--sigma", type=float, default=0.01)
    bench.add_argument("--residual-mode", choices=["l2", "affine", "affine_per_channel"], default="affine")
    bench.add_argument("--output", default="results/receipts/gates-0-1-v0.json")
    bench.add_argument("--tv-iters", type=int, default=250)
    g2 = sub.add_parser("gate2", help="Corner-geometry funnel + calibrated temperature (preregistered)")
    g2.add_argument("--seeds", type=int, nargs="+", default=None)
    g2.add_argument("--output", default="results/receipts/gate2-v0.json")
    funnel = sub.add_parser("funnel", help="What one photo's boundary decides about the corner geometry")
    funnel.add_argument("image")
    funnel.add_argument("--edge", choices=["right", "left", "top", "bottom"], default="right")
    funnel.add_argument("--region", type=float, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"),
                        help="boundary region as image fractions")
    funnel.add_argument("--output", default=None, help="write the funnel receipt as JSON")
    outpaint = sub.add_parser("outpaint",help="Generate an image extension outside the photograph")
    outpaint.add_argument("image")
    outpaint.add_argument("--output",required=True,help="Output PNG path")
    outpaint.add_argument("--guided-output",default=None,help="Also try experimental light guidance and save its comparison")
    outpaint.add_argument("--edge",choices=["right","left","top","bottom"],default="right")
    outpaint.add_argument("--extend",type=float,default=0.45)
    outpaint.add_argument("--steps",type=int,default=30)
    outpaint.add_argument("--seed",type=int,default=42)
    outpaint.add_argument("--max-side",type=int,default=512)
    outpaint.add_argument("--prompt",default=None)
    outpaint.add_argument("--model",default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    if args.command == "gui":
        from .gui import launch_gui
        launch_gui(share=args.share, server_port=args.port)
        return 0
    if args.command == "benchmark":
        from .benchmark import run_benchmark
        run_benchmark(
            args.seeds,
            candidates=args.candidates,
            sigma=args.sigma,
            residual_mode=args.residual_mode,
            output=Path(args.output),
            tv_iters=args.tv_iters,
        )
        print(f"wrote {args.output}")
        return 0
    if args.command == "gate2":
        from .gate2 import HELD_OUT_SEEDS, run_gate2
        receipt = run_gate2(args.seeds or HELD_OUT_SEEDS, output=Path(args.output))
        for name, g in receipt["result"]["gates"].items():
            print(f"{name}: {g['observed']} (required {g['required']}) -> {'pass' if g['pass'] else 'FAIL'}")
        print(f"Gate 2 {'PASSED' if receipt['result']['passed'] else 'FAILED'}; wrote {args.output}")
        return 0
    if args.command == "funnel":
        import json
        from PIL import Image
        from .photo import PhotoConfig, funnel_photo
        cfg = PhotoConfig(edge=args.edge) if args.region is None else PhotoConfig(edge=args.edge, region_fraction=tuple(args.region))
        with Image.open(args.image) as image:
            result = funnel_photo(image.convert("RGB"), cfg)
        print(result.verdict)
        for row in result.ledger:
            print(f"  level {row['level']}: tested {row['tested']}, refuted {row['refuted']}, "
                  f"best reduced chi2 {row['birge']:.2f}")
        if args.output:
            from .receipts import write_receipt
            write_receipt(result.summary(), Path(args.output))
            print(f"wrote {args.output}")
        return 0
    if args.command == "outpaint":
        from PIL import Image
        from .outpainting import DEFAULT_MODEL, OutpaintConfig, generate_outpaintings
        from .photo import PhotoConfig
        config = OutpaintConfig(edge=args.edge,extension_fraction=args.extend,steps=args.steps,seed=args.seed,
            max_side=args.max_side,prompt=args.prompt or OutpaintConfig().prompt,model_id=args.model or DEFAULT_MODEL,
            light_guidance=args.guided_output is not None)
        with Image.open(args.image) as image:
            result = generate_outpaintings(image.convert("RGB"),PhotoConfig(edge=args.edge),config)
        output = Path(args.output)
        output.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(result.prior).save(output,format="PNG")
        print(f"wrote {output}")
        if args.guided_output is not None:
            guided_output = Path(args.guided_output)
            guided_output.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(result.guided).save(guided_output,format="PNG")
            print(f"wrote {guided_output} (light guidance: {result.guidance_status})")
        for warning in result.warnings:
            print(warning)
        return 0
    raise NotImplementedError(f"{args.command} is not implemented yet")
