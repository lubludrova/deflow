# E05 — joint Anderson tolerance 1e-5, cap 10

All three fresh seeds completed training and their 1000-episode stochastic evaluations with exit 0. Each final checkpoint and all eight periodic/final checkpoints per seed were loaded remotely and checked for finite actor/critic/temperature values. Local SHA-256 matches the remote verification receipt and final sync acknowledgment.

| Seed | Real transitions | U4 | Success (%) | Return | Training SPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 200000 | 0.396386 | 56.30 | -9.14355 | 32.549 |
| 2 | 200001 | 0.315567 | 43.00 | -10.36905 | 32.056 |
| 3 | 200002 | 0.394195 | 52.60 | -10.16988 | 32.217 |
| Mean ± sample SD | — | 0.3687 ± 0.0460 | 50.6333 ± 6.8646 | -9.8942 ± 0.6576 | 32.2742 ± 0.2514 |

Sample SD uses n−1 across three training seeds. Evaluation seed is 26152. U4, success and return were independently recomputed from all 1000 episode records per seed. No seed or episode was filtered.

Training SPS = actual real transitions / DONE.elapsed_seconds. This includes unchanged diagnostics, checkpoint writing and intermediate evaluation, and excludes the final stochastic evaluation. It is end-to-end training throughput, not isolated solver performance. All seeds used RTX5090 with E02 PyTorch 2.7.1+cu128 and strict FP32. Each seed trained concurrently with its E04 cap100 peer. Shared-GPU timing is recorded in launch and exit receipts. Final E05 evaluations waited for E04 completion and GPU availability. Historical E01 shared RTX4090 timings are not a controlled speed comparator.

Protocol: MultiGoalDockTight-v0, 200000 requested transitions, num_envs=8, batch=1024, learning_starts=4000, utd_ratio=0.5. Native joint Anderson tolerance1e-5/cap10, m5/lam1e-2/beta0.7 and unchanged shared density/backward. This condition is separate from sequential Anderson and tighter tolerance1e-6/cap100.

Evidence: remote-verification.json, metrics.json, per_seed.csv, ../FROZEN.json, LAUNCH_VERIFIED.json, ops/manifest.json, ops/COMMANDS.txt and ops/FINAL_SYNC_ACK.json. Full raw solver traces/capsules stay on the server. Nothing was deleted, retried or resumed.

Operational note: preflight configuration, smoke training, smoke evaluation and shared-placement gates passed. Two additional tests checked peer identity admission and the final ACK guard. No scientific retries or resume.

## Final operational clarification — 2026-11-12T13:45:25.630771+00:00

These results belong exclusively to the fresh `restart1` campaign explicitly authorized after the original attempts were stopped by the global stop request. The original stopped attempts remain separate. There were no scientific retries or resumes within restart1. Shared-GPU throughput is not an isolated solver timing comparison. Final critical artifacts were verified locally against the server-generated verification and final synchronization receipts. See `../ops/COMPLETED.json`.

Seeds 1 and 3 encountered `Peer identity changed` in supervisor admission before their first final evaluation. The separate `evaluation-handoff-v1` performed each first evaluation once with the exact frozen evaluation command after peer completion. Original administrative failure markers remain preserved. All training and evaluation child processes exited 0. The frozen original status reader still shows those historical administrative failures. This completion receipt and the recovered evaluator receipts establish the final state.
