"""Train and evaluate the native-continuing PegInsertSide protocol."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import run_manipulation_stage1 as stage

PROTOCOL = "peg_native_continuing_v1"
RULES = {"divisor": 1.0, "success_bonus": 100.0, "success_reward": 0.0,
         "horizon": 200, "terminal_bonus": 0.0, "terminate_on_success": False}
LOCK = HERE / "deflow_mog_remote_runtime_lock.json"
METHODS = {"deflow": ("r1", "gated"), "sac": ("sacsoft", "gaussian"),
           "sacflow": ("sacflowsoft", "parent"), "dime": ("dimesoft", "dime")}
SOURCE_FILES = (
    "repro/run_peg_native_continuing.py",
    "repro/run_manipulation_stage1.py",
    "repro/run_manipulation_baseline.py", "repro/deflow_mog_remote_runtime_lock.json",
    "sac_anderson_flow.py", "sac_flow_parent.py", "metaworld_soft_env.py",
    "sac_gaussian_baseline.py", "dime_actor.py",
    "maniskill_env.py",
    "analysis/evaluate_stage1_panel.py", "analysis/cluster_ci.py",
)


def run_id(seed: int, smoke: bool = False, method: str = "deflow") -> str:
    prefix = "peg_smoke" if smoke else "peg"
    recipe, _ = METHODS[method]
    return f"{prefix}_{method}_{seed}"


def plan(seed: int, smoke: bool = False, method: str = "deflow") -> dict:
    recipe, actor = METHODS[method]
    return {
        "protocol_version": PROTOCOL, "run_id": run_id(seed, smoke, method),
        "seed": seed, "method": method, "recipe": recipe, "actor": actor,
        "total_timesteps": 2048 if smoke else 500000,
        "fresh": True, "resume": False, "rules": dict(RULES),
        "recipe_constants": dict(stage.RECIPES[recipe]), "precision": "high",
        "training_reward": "native dense; no added success bonus",
        "timeout_bootstrap": True, "autoreset": "NextStep",
        "native_success_metric": "charts/eval_native_success_rate_det",
        "legacy_success_proxy_threshold": 9.5,
    }


def runtime_check(require_match: bool) -> dict:
    lock = json.loads(LOCK.read_text())
    observed = {key: importlib.metadata.version(key) for key in lock["packages"]}
    mismatches = {key: {"expected": value, "observed": observed[key]}
                  for key, value in lock["packages"].items() if observed[key] != value}
    if platform.python_version() != lock["python"]:
        mismatches["python"] = {"expected": lock["python"], "observed": platform.python_version()}
    if stage.torch.version.cuda != lock["torch_cuda"]:
        mismatches["torch_cuda"] = {"expected": lock["torch_cuda"], "observed": stage.torch.version.cuda}
    if require_match and mismatches:
        raise RuntimeError(f"Runtime differs from {LOCK}: {json.dumps(mismatches)}")
    return {"packages": observed, "python": platform.python_version(),
            "platform": platform.platform(), "torch_cuda": stage.torch.version.cuda,
            "lock_sha256": stage.file_sha256(LOCK), "lock_mismatches": mismatches}


@contextmanager
def native_protocol():

    old_rules = stage.STAGE3_ENV_RULES["mw_peg"]
    old_spec = stage.STAGE3_SPECS["mw_peg"]
    stage.STAGE3_ENV_RULES["mw_peg"] = dict(RULES)
    stage.STAGE3_SPECS["mw_peg"] = {**old_spec, "success_threshold": 9.5}
    try:
        yield
    finally:
        stage.STAGE3_ENV_RULES["mw_peg"] = old_rules
        stage.STAGE3_SPECS["mw_peg"] = old_spec


def provenance() -> dict:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              text=True, capture_output=True)
    sources = {name: stage.file_sha256(ROOT / "src" / name) for name in SOURCE_FILES}
    source_digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    return {"code_sha": f"sha256:{source_digest}",
            "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
            "source_sha256": sources,
            "launcher_argv": sys.argv[:], "runtime": runtime_check(False)}


def train(seed: int, smoke: bool, max_seconds: int, method: str = "deflow") -> None:

    recipe, _ = METHODS[method]
    identity = run_id(seed, smoke, method)
    destination = ROOT / "runs" / identity
    if destination.exists():
        raise FileExistsError(f"Refusing to reuse {destination}")
    runtime_check(True)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu or "," in gpu or not stage.torch.cuda.is_available():
        raise RuntimeError("Pin one free GPU with CUDA_VISIBLE_DEVICES; CUDA is required.")
    if stage.torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible GPU is required.")
    receipt = provenance()
    receipt["gpu"] = stage.torch.cuda.get_device_name(0)
    original_json = stage.campaign.atomic_json
    original_argv, original_cwd = sys.argv[:], Path.cwd()
    snapshot_written = False

    def write_receipt(path, payload):
        nonlocal snapshot_written
        if not snapshot_written:
            destination.mkdir(parents=True, exist_ok=False)
            for name in SOURCE_FILES:
                target = destination / "source_snapshot" / "src" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / "src" / name, target)
            snapshot_written = True
        payload = {**payload, **receipt,
                   "schema": PROTOCOL, "protocol_version": PROTOCOL,
                   "scientific_role": "integration_smoke" if smoke else "training",
                   "reward_contract": "native dense reward; subtract inner +100 on every successful step; no success termination; H200 timeout bootstraps",
                   "soft_protocol": None, "native_continuing_protocol": dict(RULES),
                   "legacy_success_proxy_threshold": 9.5,
                   "protocol": "native_continuing"}
        original_json(path, payload)

    stage.campaign.atomic_json = write_receipt
    sys.argv = [str(HERE / "run_manipulation_stage1.py"), recipe, "mw_peg",
                "--run-id", identity, "--protocol", "soft", "--total-timesteps", str(2048 if smoke else 500000),
                "--max-seconds", str(max_seconds), "--host", socket.gethostname(),
                "--gpu", gpu, "--precision", "high"] + (["--smoke"] if smoke else [])
    try:
        os.chdir(ROOT)
        with native_protocol():
            stage.main()
    finally:
        stage.campaign.atomic_json = original_json
        sys.argv = original_argv
        os.chdir(original_cwd)


def validate_checkpoint(checkpoint: Path, metadata: dict) -> None:
    saved = stage.torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = saved.get("args", {})
    if args.get("run_name") != metadata["run_id"] or args.get("exp_name") != metadata["run_id"]:
        raise ValueError("Checkpoint identity does not match the supplied contract.")
    if saved.get("global_step") == metadata["total_timesteps"]:
        expected = metadata.get("final_checkpoint_sha256")
        if expected is None or stage.file_sha256(checkpoint) != expected:
            raise ValueError("Final checkpoint SHA does not match the completed run contract.")
    for name in SOURCE_FILES:
        if stage.file_sha256(ROOT / "src" / name) != metadata["source_sha256"].get(name):
            raise ValueError(f"Evaluation source drift: {name}; use the run's source snapshot.")


def evaluate(checkpoint: Path, contract: Path, out: Path, device: str, environments: int, samples_per_environment: int) -> None:

    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    metadata = json.loads(contract.read_text())
    if metadata.get("protocol_version") != PROTOCOL:
        raise ValueError("This evaluator requires a native-continuing run contract.")
    validate_checkpoint(checkpoint, metadata)
    if device == "cuda":
        runtime_check(True)

    stage.torch.backends.cuda.matmul.allow_tf32 = True
    stage.torch.backends.cudnn.allow_tf32 = True
    stage.torch.set_float32_matmul_precision("high")
    sys.path.insert(0, str(ROOT / "src" / "analysis"))
    import evaluate_stage1_panel as evaluator
    original_argv = sys.argv[:]
    sys.argv = [evaluator.__file__, "--env", "mw_peg", "--checkpoint", str(checkpoint),
                "--contract", str(contract), "--channels", "det,stoch", "--horizon", "200",
                "--device", device, "--out", str(out), "--environments", str(environments), "--samples-per-environment", str(samples_per_environment)]
    try:
        with native_protocol():
            evaluator.main()
        payload = json.loads(out.read_text())
        payload.update(protocol_version=PROTOCOL, reward_contract="native dense, no +100",
                       precision={"float32_matmul_precision": stage.torch.get_float32_matmul_precision(),
                                  "cuda_matmul_allow_tf32": stage.torch.backends.cuda.matmul.allow_tf32,
                                  "cudnn_allow_tf32": stage.torch.backends.cudnn.allow_tf32},
                       native_continuing_launcher_sha256=stage.file_sha256(__file__),
                       metaworld_wrapper_sha256=stage.file_sha256(ROOT / "src/metaworld_soft_env.py"),
                       runtime=runtime_check(False))
        stage.campaign.atomic_json(out, payload)
    finally:
        sys.argv = original_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "train"):
        child = commands.add_parser(name)
        child.set_defaults(seed=1)
        child.add_argument("--method", choices=tuple(METHODS), default="deflow")
        child.add_argument("--smoke", action="store_true")
        if name == "train":
            child.add_argument("--max-seconds", type=int, default=86400)
    child = commands.add_parser("eval")
    child.add_argument("--checkpoint", type=Path, required=True)
    child.add_argument("--contract", type=Path, required=True)
    child.add_argument("--out", type=Path, required=True)
    child.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    child.add_argument("--environments", type=int, default=20)
    child.add_argument("--samples-per-environment", type=int, default=5)
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(plan(args.seed, args.smoke, args.method), indent=2))
    elif args.command == "train":
        train(args.seed, args.smoke, args.max_seconds, args.method)
    else:
        evaluate(args.checkpoint, args.contract, args.out, args.device, args.environments, args.samples_per_environment)


if __name__ == "__main__":
    main()
