# E02 runner handoff — sequential Anderson ×3

Authorization: user requested the new baseline, verification and immediate three-seed launch on 12 November 2026. Read ../../README.md and the inherited rebuttal contract. This is separate from E01 at port 18290. Never replace its active profile or touch its frozen sources/coordinator.

## Launch verified

At 10:04 Asia/Shanghai on 2026-11-12 all three jobs were TRAINING beyond 5000 real transitions. This is a dated snapshot, not current state. Query the server before reporting progress.

- sequential_anderson_s1: GPU 0, PID 4058, tmux deflow-sa-20261112-s1.
- sequential_anderson_s2: GPU 1, PID 4063, tmux deflow-sa-20261112-s2.
- sequential_anderson_s3: GPU 2, PID 4061, tmux deflow-sa-20261112-s3.

Host: workspace.featurize.cn, SSH port 13534, user featurize.
Remote root: /home/featurize/deflow-sequential-anderson-20261112.
Manifest SHA256: 31f4c2449287141d788de2f9e00ac732b335e64676bf668ad8247fe8c4b932f5.
All 30 frozen source/config files verified on the server before admission.
Each job has a separate attempt 0000, atomic claim, process identity and real child exit receipt. Do not submit again, restart, resume or change the scientific settings.

## Queue and acceptance

GPU 0: S1 training → verified checkpoint sync → evaluation 1000.
GPU 1: S2 training → verified checkpoint sync → evaluation 1000.
GPU 2: S3 training → verified checkpoint sync → evaluation 1000.
Then verify all three evaluations and final local sync acknowledgment.
No additional training, no new nine-audit queue. Existing E01 audits stay with E01.

Settings: 200000 transitions, seeds 1/2/3, strict FP32, native Anderson kernel per local implicit step, cap 25 / tol 1e-5, shared E01 density and backward. Final evaluation is stochastic coverage, 1000 episodes, eval_seed 26152. Intermediate eval is deterministic.

Production configuration dictionaries and nine engine hashes match E01, apart from seed/run names and the requested forward-solver dispatch. RTX5090 requires torch 2.7.1+cu128 here, versus E01 torch 2.5.1+cu124. Report runtime/hardware mismatch and do not turn historical shared-GPU elapsed into a clean solver speed ratio.

## Status and sync

From /Users/ruslanageev/dev/DEFLOW:

```bash
python3 rebuttal/experiments/E02-sequential-anderson/runs/sa-20261112/ops/status_verified.py
launchctl list com.codex.deflow-sa-20261112-sync
```

Use status_verified.py, not the original frozen status.py. At seed 2 spawn, the initial /proc cmdline read was empty. The stable PID/start ticks/boot ID and exact contracted argv were verified after launch. ops/identity_reconciliation_s2.json preserves the evidence, and status_overlay.json pins the read-only corrected reader. No process, frozen source or original receipt was changed. The overlay only accepts this specific empty-cmdline case after matching boot, start ticks and exact manifest argv. Other mismatches remain unresolved.

Local launchd sync runs every 120 s. Its plist and stdout/stderr are in ops/. The current SSH control socket is /tmp/deflow-sa-20261112.sock, managed by task-owned SSH master PID 30863. Keep it for synchronization. No passwords were stored. If transport is lost, reauthenticate interactively using the user's server credentials and the profile's socket. Training continues in tmux. Missing/stale verified sync blocks subsequent evaluation, not training.

Existing E01 keepawake and sync were preserved. The new launchd service is bootstrapped for the current user session. After a host reboot/login, verify it is loaded and reload the saved plist if needed. Do not create a second supervisor or sync service.

The compact mirror includes checkpoints, TensorBoard events, summaries, terminal receipts and validation. Full solver_calls and training capsules remain remote. No remote deletion. Final acceptance requires real success exits, actual horizon 200000..200007, finite saved actor/q1/q2/log_alpha, source/seed/checkpoint checks, prescribed evaluation and verified sync. The final local acknowledgment is ops/FINAL_SYNC_ACK.json. Inspect errors rather than retrying training.

## Validation evidence

27 solver/integration tests + 4 supervisor tests passed. CUDA smoke training 256 + 10-episode evaluation passed. Production-batch pilot 8002 completed in 341.18 s, with finite losses/weights and strict FP32 through backward. RELEASE_READY.json records all gates, including all three GPU devices.

Fixed-capsule diagnostics used all 1024 rows of A2 at 99336 and N3 at 78398. All solver/readout/gradient values were finite. Sequential Anderson fresh-failure counts were 287/1024 and 838/1024. Reference-valid counts were 1022/1024 and 468/1024. This is implementation validation, not all-sample convergence certification. Reports and raw masks remain in validation/fixed-capsules-v3, mirrored locally. Earlier failed diagnostic attempts are retained.

Read ../../analysis/validation.md for scientific scope and preflight corrections. Production runs were fresh and have not been retried. Final learning results are pending.


## Completion verified 2026-11-12

All three scientific attempts and final 1000-episode evaluations exited 0. Frozen sources, actual horizons 200002/200006/200003, checkpoint hashes, evaluation hashes and local FINAL_SYNC_ACK were checked against the live server. Full raw stays remote and the cancelled full transfer remains cancelled.

| Seed | Final return | U4 | Success |
| --- | ---: | ---: | ---: |
| sequential_anderson_s1 | -8.19336 | 0.557025 | 86.9% |
| sequential_anderson_s2 | -7.82613 | 0.667745 | 97.7% |
| sequential_anderson_s3 | -9.04952 | 0.634373 | 91.8% |

Evidence: [completion receipt](../../analysis/completion-20261112T065541Z/VERIFIED.json), with remote exits/hashes, status snapshot and final sync acknowledgment. No further E02 jobs are queued. GPU0/1/2 were idle with no compute processes at verification.
