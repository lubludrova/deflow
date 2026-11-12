# E05 restart1

Latest user authorization explicitly supersedes the prior global STOP for these
six E04/E05 jobs only. See AUTHORIZATION.json and common/operations restart receipt.
This is a fresh replacement from step0, not exact checkpoint continuation.
Stopped original attempts and all artifacts are preserved.

Science unchanged: joint Anderson tol1e-5 cap10, seeds1/2/3, 200k real
transitions, strict FP32, matched docking config, E02 RTX5090 runtime.
Each GPU shares E04/E05 matched seeds. No additional retries or resume.
Every25k checkpoints, verified compact sync, stochastic eval1000 seed26152.
E04 final ACK waits for E05 training exits. E05 evaluation waits for E04 terminal
and GPU availability. Do not start original campaigns or bypass this gate.

Remote root: /home/featurize/deflow-joint-anderson-a10-20261112-restart1
Profile and commands: ops/profile.json and ops/manifest.json.
Status: ops/monitor_state.json and monitor_history.jsonl.
All final statistics must use these new fresh attempts. Preserve administrative
stops as separate provenance. Shared GPU SPS is not exclusive solver timing.
