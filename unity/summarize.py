#!/usr/bin/env python3
"""Summarise the Unity benchmark CSVs (drops_<collider>.csv, walks_<collider>.csv) into one table.

  python unity/summarize.py <folder with the CSVs> [--json out.json]
"""
import argparse
import csv
import glob
import json
import os

import numpy as np


def read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--tol", type=float, default=0.05, help="rest-height tolerance vs the visible surface (m)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = {}
    names = sorted({os.path.basename(p)[6:-4] for p in glob.glob(os.path.join(a.folder, "drops_*.csv"))} |
                   {os.path.basename(p)[6:-4] for p in glob.glob(os.path.join(a.folder, "walks_*.csv"))})
    for n in names:
        r = {}
        dp = os.path.join(a.folder, f"drops_{n}.csv")
        if os.path.exists(dp):
            d = read(dp)
            fell = np.array([int(x["fell_through"]) for x in d])
            ov = np.array([int(x["started_overlapping"]) for x in d])
            rest = np.array([float(x["rest_minus_ref"]) for x in d])
            drift = np.array([float(x["drift_xz"]) for x in d])
            ok = (fell == 0) & (drift < 0.05)          # rest error only means something where it did not roll away
            r["drops"] = dict(n=len(d), fell_through=float(fell.mean()), started_overlapping=float(ov.mean()),
                              rest_within_tol=float((np.abs(rest[ok]) <= a.tol).mean()) if ok.any() else None,
                              rest_above_tol=float((rest[ok] > a.tol).mean()) if ok.any() else None,
                              rest_below_tol=float((rest[ok] < -a.tol).mean()) if ok.any() else None,
                              rest_abs_p50=float(np.median(np.abs(rest[ok]))) if ok.any() else None,
                              rolled_away=float((drift >= 0.05)[fell == 0].mean()))
        wp = os.path.join(a.folder, f"walks_{n}.csv")
        if os.path.exists(wp):
            w = read(wp)
            r["walks"] = {k: float(np.mean([int(x[k]) for x in w])) for k in ("reached", "stuck", "fell_through")}
            r["walks"]["n"] = len(w)
        out[n] = r
    print(f"{'collider':>10} | {'drops: fell':>11} {'rest ok':>8} {'above':>6} {'below':>6} | {'walks: reached':>14} {'stuck':>6} {'fell':>6}")
    for n, r in out.items():
        d, w = r.get("drops", {}), r.get("walks", {})
        f = lambda x: "  -   " if x is None else f"{100 * x:5.1f}%"
        print(f"{n:>10} | {f(d.get('fell_through')):>11} {f(d.get('rest_within_tol')):>8} {f(d.get('rest_above_tol')):>6} "
              f"{f(d.get('rest_below_tol')):>6} | {f(w.get('reached')):>14} {f(w.get('stuck')):>6} {f(w.get('fell_through')):>6}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
