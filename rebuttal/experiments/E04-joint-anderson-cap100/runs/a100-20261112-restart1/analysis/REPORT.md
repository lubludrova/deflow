# E04 — joint Anderson tolerance 1e-5, cap 100

All three fresh seeds completed training and their 1000-episode stochastic evaluations with exit 0. Each final checkpoint and all eight periodic/final checkpoints per seed were loaded remotely and checked for finite actor/critic/temperature values. Local SHA-256 matches the remote verification receipt and final sync acknowledgment.

| Seed | Real transitions | U4 | Success (%) | Return | Training SPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 200006 | 0.546270 | 85.40 | -8.44667 | 13.870 |
| 2 | 200004 | 0.671016 | 97.70 | -7.54356 | 12.888 |
| 3 | 200002 | 0.663531 | 97.10 | -8.07826 | 12.599 |
| Mean ± sample SD | — | 0.6269 ± 0.0700 | 93.4000 ± 6.9347 | -8.0228 ± 0.4541 | 13.1187 ± 0.6661 |

Sample SD uses n−1 across three training seeds. Evaluation seed is 26152. U4, success and return were independently recomputed from all 1000 episode records per seed. No seed or episode was filtered.

Training SPS = actual real transitions / DONE.elapsed_seconds. This includes unchanged diagnostics, checkpoint writing and intermediate evaluation, and excludes the final stochastic evaluation. It is end-to-end training throughput, not isolated solver performance. All seeds used RTX5090 with E02 PyTorch 2.7.1+cu128 and strict FP32. After initial exclusive operation the user authorized concurrent E05 cap10 training on each GPU. Full-run SPS includes these shared-GPU intervals, recorded in the E05 launch and exit receipts. Historical E01 shared RTX4090 timings are not a controlled speed comparator.

Protocol: MultiGoalDockTight-v0, 200000 requested transitions, num_envs=8, batch=1024, learning_starts=4000, utd_ratio=0.5. Native joint Anderson tolerance1e-5/cap100, m5/lam1e-2/beta0.7 and unchanged shared density/backward. This condition is separate from sequential Anderson and tighter tolerance1e-6/cap100.

Evidence: remote-verification.json, metrics.json, per_seed.csv, ../FROZEN.json, LAUNCH_VERIFIED.json, ops/manifest.json, ops/COMMANDS.txt and ops/FINAL_SYNC_ACK.json. Full raw solver traces/capsules stay on the server. Nothing was deleted, retried or resumed.

Operational note: the initial preflight report encountered a package import-path error after successful smoke training and evaluation. A separate finishing script verified those preserved artifacts and all three GPUs without repeating scientific work. Production supervisors and scientific sources were frozen before launch.

## Final operational clarification — 2026-11-12T13:45:25.630771+00:00

These results belong exclusively to the fresh `restart1` campaign explicitly authorized after the original attempts were stopped by the global stop request. The original stopped attempts remain separate. There were no scientific retries or resumes within restart1. Shared-GPU throughput is not an isolated solver timing comparison. Final critical artifacts were verified locally against the server-generated verification and final synchronization receipts. See `../ops/COMPLETED.json`.

For restart1, E04 trained with E05 peers during their overlapping training interval, then continued after those peers finished. The earlier initial-exclusive-operation wording describes the preceding campaign history, not a controlled timing phase of restart1.
