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

## Final operational clarification — 2026-11-12T13:45:25.630771+00:00

These results belong exclusively to the fresh `restart1` campaign explicitly authorized after the original attempts were stopped by the global stop request. The original stopped attempts remain separate. There were no scientific retries or resumes within restart1. Shared-GPU throughput is not an isolated solver timing comparison. Final critical artifacts were verified locally against the server-generated verification and final synchronization receipts. See `../ops/COMPLETED.json`.

Seeds 1 and 3 encountered `Peer identity changed` in supervisor admission before their first final evaluation. The separate `evaluation-handoff-v1` performed each first evaluation once with the exact frozen evaluation command after peer completion. Original administrative failure markers remain preserved. All training and evaluation child processes exited 0. The frozen original status reader still shows those historical administrative failures. This completion receipt and the recovered evaluator receipts establish the final state.

Final results: `analysis/REPORT.md`.

Final follow-up: at 13:44:46 UTC all three server GPUs were idle (0% utilization, 2 MiB each), with no compute processes. The existing SSH control connection succeeded. All five restart1-only local monitoring/synchronization/keepawake services were unloaded after verification. Files, remote artifacts, shared SSH and unrelated services were preserved. See `ops/FINAL_OBSERVATION_AND_CLEANUP.json`.
