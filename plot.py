"""Plot learning curves and the reward ablation from runs/*/progress.csv.

    python plot.py                       # every run under runs/
    python plot.py --runs runs/ctrl0_s1 runs/ctrl_high_s1
    python plot.py --metric x_distance   # compare on a weight-independent metric

Runs whose directory name differs only by the seed suffix (``_s1`` / ``_s2``)
are grouped into one curve: mean across seeds, shaded min-max band.
"""

import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = ("episodic_return", "x_distance", "action_energy", "episodic_length",
           "forward_reward", "ctrl_cost")


def read_run(path: str):
    steps, values = defaultdict(list), defaultdict(list)
    with open(os.path.join(path, "progress.csv")) as f:
        for row in csv.DictReader(f):
            for m in METRICS:
                if row.get(m) not in (None, ""):
                    steps[m].append(int(row["global_step"]))
                    values[m].append(float(row[m]))
    return steps, values


def binned(x, y, n_bins: int, x_max: int):
    """Average y within n_bins equal-width bins of the step axis."""
    edges = np.linspace(0, x_max, n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    centres, means = [], []
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            centres.append(0.5 * (edges[b] + edges[b + 1]))
            means.append(np.mean(np.asarray(y)[sel]))
    return np.asarray(centres), np.asarray(means)


def group_name(run_dir: str) -> str:
    return re.sub(r"_s\d+$", "", os.path.basename(run_dir.rstrip("/")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=None)
    p.add_argument("--root", default="runs")
    p.add_argument("--metric", default="episodic_return", choices=METRICS)
    p.add_argument("--bins", type=int, default=60)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    runs = args.runs or sorted(
        os.path.join(args.root, d) for d in os.listdir(args.root)
        if os.path.exists(os.path.join(args.root, d, "progress.csv"))
    )
    if not runs:
        raise SystemExit(f"no runs with progress.csv under {args.root}/")

    groups = defaultdict(list)
    for r in runs:
        groups[group_name(r)].append(r)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, paths in sorted(groups.items()):
        curves, x_max = [], 0
        for path in paths:
            steps, values = read_run(path)
            if args.metric not in steps:
                continue
            x_max = max(x_max, max(steps[args.metric]))
            curves.append((steps[args.metric], values[args.metric]))
        if not curves:
            continue
        resampled = [binned(x, y, args.bins, x_max) for x, y in curves]
        length = min(len(c[1]) for c in resampled)
        xs = resampled[0][0][:length]
        ys = np.stack([c[1][:length] for c in resampled])
        mean = ys.mean(axis=0)
        ax.plot(xs, mean, label=f"{name} (n={len(curves)})", linewidth=1.8)
        if len(curves) > 1:
            ax.fill_between(xs, ys.min(axis=0), ys.max(axis=0), alpha=0.18)

    ax.set_xlabel("environment steps")
    ax.set_ylabel(args.metric.replace("_", " "))
    ax.set_title(f"Walker2d-v5 — {args.metric.replace('_', ' ')}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = args.out or f"curve_{args.metric}.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)

    # final-performance table: last 10% of each run
    print(f"\n{'run':28s} {'final ' + args.metric:>18s}")
    for name, paths in sorted(groups.items()):
        finals = []
        for path in paths:
            steps, values = read_run(path)
            if args.metric not in values:
                continue
            v = np.asarray(values[args.metric])
            finals.append(v[int(0.9 * len(v)):].mean())
        if finals:
            print(f"{name:28s} {np.mean(finals):10.1f} ± {np.std(finals):6.1f}")


if __name__ == "__main__":
    main()
