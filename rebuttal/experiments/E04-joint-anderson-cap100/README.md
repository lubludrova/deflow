> **Current attempt:** [`a100-20261112-restart1`](runs/a100-20261112-restart1/HANDOFF.md), three seeds verified TRAINING on 12 November 2026. Use its `ops/status.py` and existing sync services. [Published log snapshot](runs/a100-20261112-restart1/publication-snapshot-20261112/) is dated and contains no checkpoints. The original non-restart attempt described below was stopped and is now in the ignored local archive. No final evaluation is available for the restart snapshot.

# E04 — joint Anderson tolerance 1e-5, cap 100

User-authorized separate three-seed training campaign, 12 November 2026.
Addresses dkt6 numerical training-budget comparison, D-C10/D-C11/D-C13.
E01 A100 audit results do not supply this independent training condition.

Protocol: MultiGoalDockTight-v0, 200000 real transitions, strict FP32,
seeds 1/2/3, num_envs=8, batch=1024, learning_starts=4000, utd_ratio=0.5.
Unchanged E02 package and nine E01 engine files, joint Anderson dispatch,
native tolerance 1e-5, iteration cap 100, m=5, lam=1e-2, beta=0.7.
Runtime reuses E02 PyTorch 2.7.1+cu128 on three RTX5090 at port13534.
The earlier 1e-6/cap100 trainings are a different scientific condition.

Queue: GPU0 seed1 training -> verified checkpoint sync -> stochastic eval1000.
GPU1 seed2 and GPU2 seed3 follow the same queue. Evaluation seed26152.
Checkpoints every25000 real transitions, plus final checkpoint. No retries or
resume. No deletions. All raw traces/capsules remain remote. Local sync includes
checkpoints, events, compact logs, metadata, receipts and evaluation episodes.

Frozen commands/hashes: runs/a100-20261112/ops/manifest.json.
Status: python3 rebuttal/experiments/E04-joint-anderson-cap100/runs/a100-20261112/ops/status.py
Sync: python3 rebuttal/experiments/E04-joint-anderson-cap100/runs/a100-20261112/ops/sync_local.py

Success requires real training/evaluation exits0, horizon200000..200007,
finite checkpoints, correct training/evaluation configuration, 1000 episodes,
verified checkpoint/evaluation hashes and final local sync. Training SPS includes
unchanged diagnostics and intermediate evaluation, not isolated solver time.
Status: preparation and validation in progress. See dated receipts for live evidence.

## Concurrent placement authorized after E04 launch

The user subsequently explicitly authorized E05 joint Anderson tol1e-5/cap10
seeds1/2/3 concurrently on the same physical GPUs, two trainings per GPU.
E04 scientific sources, parameters, attempts and supervisors remain unchanged.
E05/runs/a10-20261112/ops/E04_SYNC_OVERLAY_RECEIPT.json records the separate
compact-sync publisher that delays E04 final-checkpoint ACK while its E05 peer
training lacks a real exit receipt. It still copies and verifies all artifacts.
This safely preserves E04's existing exclusive-GPU evaluation admission check.
E05 evaluates only after E04 has terminated and the physical GPU is free.

E04 runtime now includes shared-GPU intervals. Report those explicitly and do
not treat full-run SPS as an exclusive-GPU cost measurement. E05 preflight and
production receipts bound these intervals. Do not reload E04's original sync
publisher while E05 training is active, because that bypasses the ACK guard.
