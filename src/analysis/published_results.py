"""Discover runs from the published results directory, without selection manifests."""

import json
from pathlib import Path


MUJOCO_ENVS = {"hopper": "hop", "walker2d": "walk", "halfcheetah": "hc",
               "ant": "ant", "humanoid": "hum"}
MUJOCO_METHODS = {"sac": "gauss", "sacflow": "flows", "dime": "dime",
                  "qsm": "qsm", "deflow": "flowe"}
DOCKING_TASKS = {"wide_loose": "Wide docking", "wide_tight": "Tight docking",
                 "narrow_tight": "Narrow navigation", "narrow_loose": "Narrow + wide docking"}
MANIPULATION_TASKS = {"pickcube": "ms_pickcube", "pushcube": "ms_pushcube",
                      "button_press_wall": "mw_button", "push_wall": "mw_pushwall",
                      "peg_insert_side": "mw_peg_native"}


def run_directories(root, pattern="*/*/seed_*"):
    runs = sorted(path for path in Path(root).glob(pattern) if path.is_dir())
    if not runs:
        raise ValueError(f"No result runs found in {root}")
    return runs


def mujoco_records(root):
    root = Path(root)
    records = []
    for run in run_directories(root):
        environment, method, seed = run.relative_to(root).parts
        records.append(dict(environment=MUJOCO_ENVS[environment],
                            method=MUJOCO_METHODS[method], seed=seed,
                            path=str((run / "curves.json.gz").relative_to(root))))
    return records


def docking_records(root):
    root = Path(root)
    records = []
    for run in run_directories(root):
        task, method, seed = run.relative_to(root).parts
        evaluations = sorted((run / "evaluations").glob("step_*.json"))
        if not evaluations:
            raise ValueError(f"No evaluations in {run}")
        last_step = json.loads(evaluations[-1].read_text())["checkpoint_global_step"]
        status_path = run / "status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text())
            if status.get("status") != "FAILED" or not isinstance(status.get("failure_transition"), int):
                raise ValueError(f"Invalid failure status in {status_path}")
            if not last_step < status["failure_transition"] < 200_000:
                raise ValueError(f"Failure step conflicts with evaluations in {run}")
        elif last_step >= 200_000:
            status = {"status": "evaluated"}
        else:
            raise ValueError(f"Incomplete run needs status.json with its recorded failure: {run}")
        records.append(dict(task=DOCKING_TASKS[task], method=method, seed=seed,
                            **status, checkpoint_checks=[
                                {"evaluation": str(path.relative_to(root))} for path in evaluations]))
    return records


def manipulation_records(root):
    root = Path(root)
    records = []
    for run in run_directories(root):
        task, method, seed = run.relative_to(root).parts
        events = sorted((run / "events").glob("events.out.tfevents.*"))
        if not events:
            raise ValueError(f"No evaluation event files in {run}")
        records.append(dict(panel=MANIPULATION_TASKS[task], method=method, seed=seed,
                            run_id=str(run.relative_to(root)),
                            events=[str(path.relative_to(root)) for path in events],
                            eval=str((run / "evaluations/step_000500000.json").relative_to(root))))
    return records


def solver_records(root):
    root = Path(root)
    records = []
    for run in run_directories(root, "*/seed_*"):
        task, seed = run.relative_to(root).parts
        records.append(dict(task=MANIPULATION_TASKS[task], seed=seed,
                            evaluations={arm: str((run / f"{arm}.json").relative_to(root))
                                         for arm in ("original", "tight100")},
                            accuracy=str((run / "accuracy.json").relative_to(root))))
    return records
