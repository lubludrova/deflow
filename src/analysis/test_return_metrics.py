"""Regression checks for the versioned paper return protocol."""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parent))
from return_metrics import (GRID, curve_summary, load_runs, mean_ci,
                            observed, read_curve, score)


def test_checkpoint_not_last_available_or_best():
    assert score(GRID, np.arange(40)) == 39
    assert score(GRID[:-1], np.arange(39)) is None
    assert score([950000, 1000000, 1025000], [100, 4, 90]) == 4


def test_no_trimming_and_fixed_tail_window():
    values = np.zeros(40)
    values[-4:] = [0, 0, 0, 100]
    assert score(GRID, values, "tail100k") == 25
    assert score(GRID[:-1], values[:-1], "tail100k") is None
    assert score(GRID, values, "final") == 100


def test_auc_uses_all_scheduled_points_with_no_extrapolation():
    assert score(GRID, GRID / 1e6, "auc") == pytest.approx(0.5125)
    assert score(GRID[1:], GRID[1:], "auc") is None
    assert np.isnan(observed([1, 3], [2, 6], [2])[0])


def test_student_t_interval_and_single_seed():
    cell = mean_ci([1, 3])
    half = t.ppf(.975, 1)
    assert cell["ci95"] == pytest.approx([2 - half, 2 + half])
    assert mean_ci([4])["ci95"] is None
    assert mean_ci([])["mean"] is None
    with pytest.raises(ValueError):
        mean_ci([1, np.nan])


def test_fixed_cohort_curve_no_smoothing_or_survivor_switch():
    grid, mean, low, high = curve_summary([([1, 2], [0, 100]), ([1], [0])], [1, 2])
    assert list(grid) == [1]
    assert list(mean) == [0]
    assert list(low) == list(high) == [0]


@pytest.mark.parametrize("nested", [False, True])
def test_both_export_schemas(tmp_path, nested):
    data = {"charts/eval_return_det": [[25000, 5], [1000000, 7]]}
    if nested:
        data = {"data": {"charts/eval_return_det": {"steps": [25000, 1000000], "values": [5, 7]}}}
    path = tmp_path / "run_s1.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump(data, stream)
    runs = load_runs(tmp_path, [{"path": path.name, "seed": 0}])
    assert runs[0]["scores"]["final"] == 7
    assert runs[0]["scores"]["auc"] is None


def test_bad_series_is_not_silently_dropped(tmp_path):
    path = tmp_path / "bad.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump({"charts/eval_return_det": [[1, 2], [1, 3]]}, stream)
    with pytest.raises(ValueError, match="non-increasing"):
        read_curve(path)


def test_duplicate_seed_cannot_inflate_sample_size(tmp_path):
    for name in ("a_s1.json.gz", "b_s1.json.gz"):
        with gzip.open(tmp_path / name, "wt") as stream:
            json.dump({"charts/eval_return_det": [[1000000, 7]]}, stream)
    with pytest.raises(ValueError, match="Duplicate replicate"):
        load_runs(tmp_path, [{"path": name, "seed": 0} for name in ("a_s1.json.gz", "b_s1.json.gz")])
