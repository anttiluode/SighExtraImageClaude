"""Gate 3: locate a wall corner from the 2-D penumbra fan on the floor.

Protocol: docs/superpowers/specs/2026-10-03-gate3-fan-apex-design.md
"""
from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path
import math
import time

from .fan import FanConfig, distance_to_line, locate_apex, render_fan_scene
from .receipts import write_receipt

HELD_OUT_SEEDS = (41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107)
PRIMARY = {"leak": 0.3, "texture": 0.01, "noise_std": 0.002}
SWEEP = ({"leak": 0.1, "texture": 0.01, "noise_std": 0.002},
         {"leak": 0.3, "texture": 0.03, "noise_std": 0.002})
BLOCK_PX = 8.0
EDGE_GAP = 0.25          # rad: two source edges this far apart give two distinct shadow lines


def distinct_edges(sources: list[dict]) -> int:
    th = sorted(s["theta"] for s in sources)
    if not th:
        return 0
    n = 1
    for a, b in zip(th, th[1:]):
        if b - a >= EDGE_GAP:
            n += 1
    return n


def _scene_record(args) -> dict:
    seed, null, setting, config = args
    t0 = time.time()
    sc = render_fan_scene(seed, null=null, **setting)
    r = locate_apex(sc.image_srgb, config, mask=sc.floor_mask, truth=sc)
    err = math.hypot(r.best.apex_x - sc.apex[0], r.best.apex_y - sc.apex[1])
    return {
        "seed": seed, "null": null, "setting": setting,
        "truth": {"apex": sc.apex.tolist(), "hidden_ccw": sc.hidden_ccw, "n_sources": len(sc.sources),
                  "distinct_edges": distinct_edges(sc.sources),
                  "sources": sc.sources, "floor_fraction": float(sc.floor_mask.mean())},
        "result": r.summary(),
        "apex_error_px": err,
        "distance_to_line_px": distance_to_line(r, sc.apex),
        "side_correct": bool(r.best.hidden_ccw == sc.hidden_ccw),
        "truth_covered_all_levels": all(l.get("truth_still_covered", False) for l in r.ledger),
        "nearest_survivor_px": r.ledger[-1].get("truth_nearest_survivor_px"),
        "seconds": round(time.time() - t0, 1),
    }


def evaluate(scenes: list[dict], nulls: list[dict]) -> dict:
    LOCATED = ("located", "located-side-undecided")
    detected = lambda r: r["result"]["status"] != "no-signal"
    false_alarms = sum(1 for r in nulls if detected(r))
    multi = [r for r in scenes if r["truth"]["distinct_edges"] >= 2]
    single = [r for r in scenes if r["truth"]["distinct_edges"] == 1]
    multi_ok = sum(1 for r in multi if r["result"]["status"] in LOCATED and r["apex_error_px"] <= BLOCK_PX)
    single_ok = sum(1 for r in single if detected(r) and r["distance_to_line_px"] <= 2 * BLOCK_PX)
    located = [r for r in scenes if r["result"]["status"] in LOCATED]
    side_ok = sum(1 for r in located if r["side_correct"])
    confident_wrong = sum(1 for r in located if r["apex_error_px"] > 2 * BLOCK_PX)
    frac = lambda a, b: (a / b) if b else 1.0
    gates = {
        "G3a_no_false_alarms": {"observed": f"{false_alarms}/{len(nulls)}", "required": "<= 1",
                                "pass": false_alarms <= 1},
        "G3b_distinct_edges_located": {"observed": f"{multi_ok}/{len(multi)}", "required": ">= 75% within 8 px",
                                       "pass": frac(multi_ok, len(multi)) >= 0.75},
        "G3c_single_edge_line": {"observed": f"{single_ok}/{len(single)}",
                                 "required": ">= 75% detected and within 16 px of the line",
                                 "pass": frac(single_ok, len(single)) >= 0.75},
        "G3d_side_when_located": {"observed": f"{side_ok}/{len(located)}", "required": ">= 90% of located",
                                  "pass": frac(side_ok, len(located)) >= 0.9},
        "G3e_no_confident_wrong": {"observed": f"{confident_wrong}/{len(located)}",
                                   "required": "0 located verdicts > 16 px off", "pass": confident_wrong == 0},
    }
    statuses = ("located", "located-side-undecided", "on-a-line", "undecided", "no-signal", "refuted-model")
    errs = sorted(r["apex_error_px"] for r in multi if r["result"]["status"] in LOCATED)
    describe = {
        "status_counts": {s: sum(1 for r in scenes if r["result"]["status"] == s) for s in statuses},
        "null_status_counts": {s: sum(1 for r in nulls if r["result"]["status"] == s) for s in statuses},
        "distinct_edge_scenes": len(multi), "single_edge_scenes": len(single),
        "located_median_error_px": errs[len(errs) // 2] if errs else None,
        "single_edge_side_correct": f"{sum(1 for r in single if detected(r) and r['side_correct'])}/"
                                    f"{sum(1 for r in single if detected(r))}",
        "null_max_background_delta": max((r["result"]["background_delta"] for r in nulls), default=None),
        "signal_min_background_delta": min((r["result"]["background_delta"] for r in scenes), default=None),
        "truth_covered_all_levels": f"{sum(1 for r in scenes if r['truth_covered_all_levels'])}/{len(scenes)}",
    }
    return {"gates": gates, "passed": all(g["pass"] for g in gates.values()), "descriptive": describe}


def run_gate3(seeds=HELD_OUT_SEEDS, *, output="results/receipts/gate3-v0.json", workers: int = 2,
              sweep: bool = True, config: FanConfig = FanConfig(), log=print) -> dict:
    seeds = [int(s) for s in seeds]
    jobs = [(s, n, PRIMARY, config) for s in seeds for n in (False, True)]
    with Pool(workers) as pool:
        recs = []
        for rec in pool.imap(_scene_record, jobs):
            recs.append(rec)
            if log:
                log(f"seed {rec['seed']} {'null' if rec['null'] else 'signal'}: {rec['result']['status']}, "
                    f"error {rec['apex_error_px']:.1f} px, line {rec['distance_to_line_px']:.1f} px, "
                    f"{rec['seconds']}s")
        scenes = [r for r in recs if not r["null"]]
        nulls = [r for r in recs if r["null"]]
        sweeps = []
        if sweep:
            for setting in SWEEP:
                sj = [(s, False, setting, config) for s in seeds]
                sr = list(pool.imap(_scene_record, sj))
                sweeps.append({"setting": setting, "scenes": sr,
                               "result": evaluate(sr, [])["descriptive"],
                               "multi_edge_located_within_8px": sum(
                                   1 for r in sr if r["truth"]["distinct_edges"] >= 2
                                   and r["result"]["status"] in ("located", "located-side-undecided")
                                   and r["apex_error_px"] <= BLOCK_PX),
                               "multi_edge_total": sum(1 for r in sr if r["truth"]["distinct_edges"] >= 2),
                               "confident_wrong": sum(1 for r in sr if r["result"]["status"] in
                                                      ("located", "located-side-undecided")
                                                      and r["apex_error_px"] > 2 * BLOCK_PX)})
                if log:
                    log(f"sweep {setting}: {sweeps[-1]['multi_edge_located_within_8px']}/"
                        f"{sweeps[-1]['multi_edge_total']} multi-edge located, "
                        f"{sweeps[-1]['confident_wrong']} confident-wrong")
    receipt = {"schema": "sighextraimage.gate3.v0",
               "protocol": "docs/superpowers/specs/2026-10-03-gate3-fan-apex-design.md",
               "seeds": seeds, "primary_setting": PRIMARY, "config": config,
               "scenes": scenes, "nulls": nulls, "result": evaluate(scenes, nulls), "sweep": sweeps}
    write_receipt(receipt, output)
    return receipt
