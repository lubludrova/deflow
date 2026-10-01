"""Compute run-level return metrics and confidence intervals."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

EVAL_KEY = "charts/eval_return_det"
HORIZON = 1_000_000
GRID = np.arange(25_000, HORIZON + 1, 25_000)
METRICS = ("final", "tail100k", "auc")
ENVS = [("hop", "Hopper"), ("walk", "Walker2d"), ("hc", "HalfCheetah"),
        ("ant", "Ant"), ("hum", "Humanoid")]
MODELS = [("gauss", "SAC", "#7f7f7f"), ("flows", "SAC-Flow", "#9467bd"),
          ("dime", "DIME", "#ff7f0e"), ("qsm", "QSM", "#2ca02c"),
          ("flowe", "DEFlow", "#d62728")]


def read_curve(path, tag=EVAL_KEY):

    with gzip.open(path, "rt") as stream:
        data = json.load(stream)
    series = data.get(tag, data.get("data", {}).get(tag))
    if series is None:
        raise ValueError(f"{path}: missing {tag}")
    if isinstance(series, dict):
        if len(series["steps"]) != len(series["values"]):
            raise ValueError(f"{path}: mismatched steps and values")
        series = list(zip(series["steps"], series["values"]))
    arr = np.asarray(series, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or not len(arr):
        raise ValueError(f"{path}: empty or malformed evaluation series")
    if not np.isfinite(arr).all() or not np.all(np.diff(arr[:, 0]) > 0):
        raise ValueError(f"{path}: non-finite values or non-increasing steps")
    return arr[:, 0], arr[:, 1]


def observed(steps, values, grid):

    lookup = dict(zip(steps, values))
    return np.array([lookup.get(step, np.nan) for step in grid], dtype=float)


def score(steps, values, metric="final", horizon=HORIZON):
    if metric == "final":
        grid = np.array([horizon])
    elif metric == "tail100k":
        grid = np.arange(horizon - 75_000, horizon + 1, 25_000)
    elif metric == "auc":
        grid = np.arange(25_000, horizon + 1, 25_000)
    else:
        raise ValueError(f"Unknown return metric: {metric}")
    vals = observed(steps, values, grid)
    if not np.isfinite(vals).all():
        return None
    if metric == "auc":
        if len(grid) < 2:
            raise ValueError("AUC requires at least two observations")
        return float(np.trapz(vals, grid) / (grid[-1] - grid[0]))
    return float(np.mean(vals))


def mean_ci(values):

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Seed values must be a finite one-dimensional array")
    n = len(values)
    if not n:
        return {"n": 0, "mean": None, "sd": None, "ci95": None}
    mean = float(values.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "sd": None, "ci95": None}
    sd = float(values.std(ddof=1))
    half = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return {"n": n, "mean": mean, "sd": sd, "ci95": [mean - half, mean + half]}


def curve_summary(curves, grid=GRID):

    if not curves:
        return np.array([]), np.array([]), np.array([]), np.array([])
    vals = np.array([observed(s, v, grid) for s, v in curves])
    mask = np.isfinite(vals).all(axis=0)
    vals = vals[:, mask]
    mean = vals.mean(axis=0)
    if len(vals) < 2:
        lo = hi = np.full_like(mean, np.nan)
    else:
        half = t.ppf(0.975, len(vals) - 1) * vals.std(axis=0, ddof=1) / np.sqrt(len(vals))
        lo, hi = mean - half, mean + half
    return np.asarray(grid)[mask], mean, lo, hi


def load_runs(root, records):
    runs = []
    identities = set()
    for record in records:
        path = Path(root) / record["path"]
        identity = record.get("replicate", record.get("seed"))
        if identity is None:
            raise ValueError("Missing replicate identity")
        if identity in identities:
            raise ValueError(f"Duplicate replicate {identity} in one cohort")
        identities.add(identity)
        steps, values = read_curve(path)
        runs.append({"source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     **({"replicate": identity} if "replicate" in record else {"seed": identity}), "last_step": int(steps[-1]),
                     "steps": steps.tolist(), "returns": values.tolist(),
                     "scores": {m: score(steps, values, m) for m in METRICS},
                     "return_950k": score(steps, values, horizon=950_000)})
    return runs


def metric_summary(runs, metric="final"):
    return mean_ci([r["scores"][metric] for r in runs if r["scores"][metric] is not None])
