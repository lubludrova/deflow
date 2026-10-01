"""Checks for direct discovery of published continuous-control evaluations."""

import gzip
import json
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ANALYSIS = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS))
spec = importlib.util.spec_from_file_location("preview", ANALYSIS / "continuous_control_preview.py")
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


def test_discovery_includes_every_run_and_ignores_local_manifest(tmp_path):
    for index in range(4):
        path = tmp_path / "hopper" / "deflow" / f"seed_{index:02d}" / "curves.json.gz"
        path.parent.mkdir(parents=True)
        with gzip.open(path, "wt") as stream:
            json.dump({"charts/eval_return_det": [[int(step), index + float(step)/1e6]
                       for step in preview.GRID]}, stream)
    (tmp_path / "manifest.json").write_text("not a valid manifest")
    cell = preview.cohort(tmp_path)["hop"]["flowe"]
    assert cell["n_runs"] == 4
    assert cell["final_values"] == [1., 2., 3., 4.]
    assert cell["final_sd"] == pytest.approx(np.std([1., 2., 3., 4.], ddof=1))


def test_incomplete_run_is_not_silently_excluded(tmp_path):
    path = tmp_path / "hopper/deflow/seed_01/curves.json.gz"
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt") as stream:
        json.dump({"charts/eval_return_det": [[975000, 42.5]]}, stream)
    with pytest.raises(ValueError, match="no final checkpoint"):
        preview.cohort(tmp_path)


def test_missing_run_data_is_not_silently_excluded(tmp_path):
    (tmp_path / "hopper/deflow/seed_01").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        preview.cohort(tmp_path)
