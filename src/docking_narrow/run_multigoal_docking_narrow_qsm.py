"""QSM addition for the narrow-navigation campaign (second-runner module)."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import run_multigoal_docking_full as base
import run_multigoal_docking_narrow as narrow

engine = base.engine
METHODS = ("qsm",)
ENVIRONMENT = narrow.ENVIRONMENT
SOURCE_FILES = base.SOURCE_FILES + (
    "run_multigoal_docking_narrow.py",
    "run_multigoal_docking_narrow_qsm.py",
    "evaluate_multigoal_docking_narrow.py",
    "evaluate_multigoal_docking_narrow_qsm.py",
    "training_diagnostics.py",
)
GEOMETRY = narrow.GEOMETRY


def source_hashes():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def configure(method: str, sigma: str, seed: int, steps: int, smoke: bool = False,
              run_id: str | None = None):
    if method not in METHODS or sigma != "0.03":
        raise ValueError("narrow QSM module requires method qsm and sigma 0.03")
    run_id = run_id or (f"docknav020_s003_{method}_s{seed}_n{steps}"
                        + ("_smoke" if smoke else ""))
    args = base.configure(method, sigma, seed, steps, smoke, run_id)
    args.env_id = ENVIRONMENT
    return args


def run_contract(method, seed, smoke=False):
    steps = 2048 if smoke else 200_000
    args = configure(method, "0.03", seed, steps, smoke)
    return args, {
        "run_id": args.run_name, "method": method, "geometry": GEOMETRY,
        "docking_sigma": 0.03, "actor": "qsm_actor.qsm_actor",
        "critic": "sac_flow_parent.parent_critic",
        "recipe": "existing tight-docking recipe; navigation geometry only; QSM score-matching addition",
        "configuration": asdict(args), "source_sha256": source_hashes(),
        "runtime": "fresh NEXT_STEP; global_step counts replay transitions",
        "diagnostics": {"native_failure_capsule": True,
                        "note": "frozen qsm_actor path; per-update gradient guard not wired"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=METHODS)
    parser.set_defaults(seed=1)
    parser.add_argument("--smoke", action="store_true", help="2048 transitions; technical check only")
    parser.add_argument("--plan", action="store_true")
    cli = parser.parse_args()
    args, contract = run_contract(cli.method, cli.seed, cli.smoke)
    if cli.plan:
        print(json.dumps(contract, indent=2, sort_keys=True))
        return
    run_dir = Path("runs") / args.run_name
    run_dir.parent.mkdir(exist_ok=True)
    run_dir.mkdir()
    with (run_dir / "run_contract.json").open("x", encoding="utf-8") as output:
        json.dump(contract, output, indent=2, sort_keys=True)
        output.write("\n")
    print(f"identity-probe: actor={contract['actor']} critic={contract['critic']} run_id={args.run_name}")
    import qsm_actor
    qsm_actor.main(args)


if __name__ == "__main__":
    main()
