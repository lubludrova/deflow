"""Check return summaries on discovered evaluation curves."""

import gzip
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from continuous_control_sensitivity import build_cells, deflow_table
from continuous_control_preview import cohort
from return_metrics import GRID, score


def test_sensitivity_uses_all_published_runs(tmp_path):
    for index in range(4):
        path = tmp_path / "hopper" / "deflow" / f"seed_{index:02d}" / "curves.json.gz"
        path.parent.mkdir(parents=True)
        with gzip.open(path, "wt") as stream:
            json.dump({"charts/eval_return_det": [[int(step), index + float(step)/1e6]
                       for step in GRID]}, stream)
    cells = build_cells(tmp_path)
    cell = cells["hop"]["flowe"]
    assert cell["summary"]["final"]["n"] == 4
    assert cell["summary"]["final"]["mean"] == cohort(tmp_path)["hop"]["flowe"]["final_mean"]
    assert cell["summary"]["at950k"]["mean"] == pytest.approx(2.45)
    assert "R4" in deflow_table(cells)


def test_missing_grid_is_not_imputed():
    assert score([950000, 1000000], [10, 20], "tail100k") is None
    assert score([950000, 1000000], [10, 20], "auc") is None
    assert score([950000, 1000000], [10, 20], horizon=950000) == 10
