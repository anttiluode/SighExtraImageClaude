"""Gate 4: certify before claiming — the corner is a point only if probing finds no other.

Protocol: docs/superpowers/specs/2026-10-04-gate4-certified-apex-design.md
"""
from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path
import math
import time

from .fan import FanConfig, CertifyConfig, certify, certified_distance_to_line, locate_apex, render_fan_scene
from .gate3 import distinct_edges
from .receipts import write_receipt

HELD_OUT_SEEDS = (109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193)
PRIMARY = {"leak": 0.3, "texture": 0.01, "noise_std": 0.002}
SWEEP = ({"leak": 0.1, "texture": 0.01, "noise_std": 0.002},
         {"leak": 0.3, "texture": 0.03, "noise_std": 0.002})
BLOCK_PX = 8.0
LOCATED = ("located", "located-side-undecided")


def _record(args) -> dict:
    seed, null, setting = args
    t0 = time.time()
    sc = render_fan_scene(seed, null=null, **setting)
    funnel = locate_apex(sc.image_srgb, FanConfig(), mask=sc.floor_mask, truth=sc)
    rec = {"seed": seed, "null": null, "setting": setting,
           "truth": {"apex": sc.apex.tolist(), "hidden_ccw": sc.hidden_ccw, "n_sources": len(sc.sources),
                     "distinct_edges": distinct_edges(sc.sources), "sources": sc.sources},
           "funnel": funnel.summary()}
    if funnel.status in ("no-signal", "refuted-model"):
        rec.update(status=funnel.status, apex_error_px=None, distance_to_line_px=None, side_correct=None)
    else:
        cert = certify(sc.image_srgb, funnel, CertifyConfig(), mask=sc.floor_mask)
        rec.update(status=cert.status, certificate=cert.summary(),
                   apex_error_px=math.hypot(cert.best.apex_x - sc.apex[0], cert.best.apex_y - sc.apex[1]),
                   distance_to_line_px=certified_distance_to_line(cert, sc.apex),
                   side_correct=bool(cert.best.hidden_ccw == sc.hidden_ccw))
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def _confident_wrong(recs) -> int:
    return sum(1 for r in recs if r["status"] in LOCATED and r["apex_error_px"] > 2 * BLOCK_PX)


def evaluate(scenes: list[dict], nulls: list[dict], sweeps: list[list[dict]]) -> dict:
    detected = lambda r: r["status"] not in ("no-signal", "refuted-model")
    frac = lambda a, b: (a / b) if b else 1.0
    false_alarms = sum(1 for r in nulls if detected(r))
    multi = [r for r in scenes if r["truth"]["distinct_edges"] >= 2]
    single = [r for r in scenes if r["truth"]["distinct_edges"] == 1]
    multi_ok = sum(1 for r in multi if r["status"] in LOCATED and r["apex_error_px"] <= BLOCK_PX)
    single_ok = sum(1 for r in single if detected(r) and r["status"] not in LOCATED
                    and r["distance_to_line_px"] <= 2 * BLOCK_PX)
    located = [r for r in scenes if r["status"] in LOCATED]
    side_ok = sum(1 for r in located if r["side_correct"])
    all_signal = scenes + [r for sw in sweeps for r in sw]
    wrong = _confident_wrong(all_signal) + sum(1 for r in nulls if r["status"] in LOCATED)
    n_located_all = sum(1 for r in all_signal if r["status"] in LOCATED)
    gates = {
        "G4a_no_false_alarms": {"observed": f"{false_alarms}/{len(nulls)}", "required": "<= 1",
                                "pass": false_alarms <= 1},
        "G4b_distinct_edges_located": {"observed": f"{multi_ok}/{len(multi)}", "required": ">= 75% within 8 px",
                                       "pass": frac(multi_ok, len(multi)) >= 0.75},
        "G4c_single_edge_line": {"observed": f"{single_ok}/{len(single)}",
                                 "required": ">= 60% detected, not located, true corner within 16 px of the line",
                                 "pass": frac(single_ok, len(single)) >= 0.60},
        "G4d_side_when_located": {"observed": f"{side_ok}/{len(located)}", "required": ">= 90% of located",
                                  "pass": frac(side_ok, len(located)) >= 0.9},
        "G4e_never_confidently_wrong": {"observed": f"{wrong} of {n_located_all} located (primary + sweeps + nulls)",
                                        "required": "0 located verdicts > 16 px off", "pass": wrong == 0},
    }
    statuses = ("located", "located-side-undecided", "on-a-line", "undecided", "no-signal", "refuted-model")
    describe = {
        "status_counts": {s: sum(1 for r in scenes if r["status"] == s) for s in statuses},
        "null_status_counts": {s: sum(1 for r in nulls if r["status"] == s) for s in statuses},
        "distinct_edge_scenes": len(multi), "single_edge_scenes": len(single),
        "located_errors_px": sorted(round(r["apex_error_px"], 2) for r in located),
        "single_edge_line_distances_px": sorted(round(r["distance_to_line_px"], 1) for r in single if detected(r)),
        "sweeps": [{"setting": sw[0]["setting"] if sw else None,
                    "status_counts": {s: sum(1 for r in sw if r["status"] == s) for s in statuses},
                    "located_within_8px": sum(1 for r in sw if r["status"] in LOCATED and r["apex_error_px"] <= BLOCK_PX),
                    "confident_wrong": _confident_wrong(sw)} for sw in sweeps],
    }
    return {"gates": gates, "passed": all(g["pass"] for g in gates.values()), "descriptive": describe}


def run_gate4(seeds=HELD_OUT_SEEDS, *, output="results/receipts/gate4-v0.json", workers: int = 2, log=print) -> dict:
    seeds = [int(s) for s in seeds]
    jobs = [(s, n, PRIMARY) for s in seeds for n in (False, True)]
    jobs += [(s, False, st) for st in SWEEP for s in seeds]
    recs = []
    with Pool(workers) as pool:
        for rec in pool.imap(_record, jobs):
            recs.append(rec)
            if log:
                err = "" if rec["apex_error_px"] is None else f", error {rec['apex_error_px']:.1f} px, line {rec['distance_to_line_px']:.1f} px"
                log(f"seed {rec['seed']} {'null' if rec['null'] else 'signal'} {rec['setting']}: {rec['status']}{err}, {rec['seconds']}s")
    scenes = [r for r in recs if not r["null"] and r["setting"] == PRIMARY]
    nulls = [r for r in recs if r["null"]]
    sweeps = [[r for r in recs if not r["null"] and r["setting"] == st] for st in SWEEP]
    receipt = {"schema": "sighextraimage.gate4.v0",
               "protocol": "docs/superpowers/specs/2026-10-04-gate4-certified-apex-design.md",
               "seeds": seeds, "primary_setting": PRIMARY, "funnel_config": FanConfig(),
               "certify_config": CertifyConfig(), "scenes": scenes, "nulls": nulls, "sweeps": sweeps,
               "result": evaluate(scenes, nulls, sweeps)}
    write_receipt(receipt, output)
    return receipt
