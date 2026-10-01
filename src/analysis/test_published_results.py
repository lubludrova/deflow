"""Keep incomplete runs visible when discovering results without manifests."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from docking_three_env_results import load_rows
from published_results import docking_records, manipulation_records, solver_records


def docking_run(root, final_step):
    run = root / "wide_tight/explicit/seed_01"
    evaluations = run / "evaluations"
    evaluations.mkdir(parents=True)
    for step in (25_000, final_step):
        (evaluations / f"step_{step:09d}.json").write_text(json.dumps({
            "method": "explicit", "seed": "seed_01", "checkpoint_global_step": step,
            "evaluation_start": [0., 0.],
            "episodes": [{"goal": 0, "success": True, "return": 2.}],
            "metrics": {"u4": .25, "total_success": 1., "mean_return": 2.},
        }))
    return run


def test_failure_status_and_partial_curve_are_preserved(tmp_path):
    run = docking_run(tmp_path, 175_000)
    (run / "status.json").write_text(json.dumps({"status": "FAILED", "failure_transition": 184191}))
    (tmp_path / "manifest.json").write_text("invalid and deliberately ignored")
    row, = load_rows(tmp_path)
    assert row["status"] == "FAILED" and row["failure_transition"] == 184191
    assert [point["step"] for point in row["curve"]] == [25_000, 175_000]


def test_incomplete_docking_run_requires_recorded_status(tmp_path):
    docking_run(tmp_path, 175_000)
    with pytest.raises(ValueError, match="Incomplete run needs status.json"):
        docking_records(tmp_path)


def test_completed_run_needs_only_evaluations(tmp_path):
    docking_run(tmp_path, 200_005)
    row, = load_rows(tmp_path)
    assert row["status"] == "evaluated"
    assert row["u4"] == .25


def test_failure_cannot_precede_last_evaluation(tmp_path):
    run = docking_run(tmp_path, 175_000)
    (run / "status.json").write_text(json.dumps({"status": "FAILED", "failure_transition": 170000}))
    with pytest.raises(ValueError, match="Failure step conflicts"):
        docking_records(tmp_path)


def test_evaluation_identity_must_match_run_directory(tmp_path):
    run = docking_run(tmp_path, 200_005)
    path = run / "evaluations/step_000200005.json"
    data = json.loads(path.read_text())
    data["method"] = "deflow"
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        load_rows(tmp_path)


def test_manipulation_run_without_events_is_not_skipped(tmp_path):
    (tmp_path / "pickcube/deflow/seed_01").mkdir(parents=True)
    with pytest.raises(ValueError, match="No evaluation event files"):
        manipulation_records(tmp_path)


def test_solver_discovery_keeps_incomplete_pair_for_validation(tmp_path):
    run = tmp_path / "peg_insert_side/seed_03"
    run.mkdir(parents=True)
    (run / "original.json").write_text("{}")
    record, = solver_records(tmp_path)
    assert record["task"] == "mw_peg_native"
    assert record["evaluations"]["tight100"] == "peg_insert_side/seed_03/tight100.json"
