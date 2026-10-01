"""Cluster-robust and Wilson confidence intervals for Stage 1 evaluations."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def wilson_interval(successes: int, total: int, z: float = 1.96) -> dict:

    if total <= 0:
        return {"low": 0.0, "high": 1.0}
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
            / denom)
    return {"low": max(0.0, centre - half), "high": min(1.0, centre + half)}


def cluster_bootstrap_interval(clusters: Sequence[Sequence[float]],
                               resamples: int = 10_000, alpha: float = 0.05,
                               seed: int = 1) -> dict:


    arrays = [np.asarray(list(c), dtype=float) for c in clusters if len(c)]
    if not arrays:
        return {"low": 0.0, "high": 1.0, "clusters": 0, "episodes": 0,
                "point": 0.0, "resamples": resamples}
    pooled = np.concatenate(arrays)
    point = float(pooled.mean())
    if len(arrays) < 2 or resamples <= 0:
        return {"low": point, "high": point, "clusters": len(arrays),
                "episodes": int(pooled.size), "point": point,
                "resamples": resamples}
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=float)
    n = len(arrays)
    for index in range(resamples):
        picks = rng.integers(0, n, size=n)
        draws[index] = float(np.concatenate([arrays[p] for p in picks]).mean())
    low, high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"low": float(low), "high": float(high), "clusters": n,
            "episodes": int(pooled.size), "point": point,
            "resamples": resamples}


def summarize_rate(episodes: Iterable[dict], key: str,
                   resamples: int = 10_000, seed: int = 1) -> dict:

    values = [(episode["env_seed"], 1.0 if episode[key] else 0.0)
              for episode in episodes]
    clusters: dict[int, list[float]] = {}
    for env_seed, value in values:
        clusters.setdefault(env_seed, []).append(value)
    successes = int(sum(value for _, value in values))
    total = len(values)
    wilson = wilson_interval(successes, total)
    bootstrap = cluster_bootstrap_interval(list(clusters.values()),
                                           resamples=resamples, seed=seed)
    return {
        "successes": successes, "episodes": total,
        "rate": successes / total if total else None,
        "wilson": wilson, "cluster_bootstrap": bootstrap,
        "clusters": len(clusters),
    }
