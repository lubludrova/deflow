> **Current attempt:** [`a10-20261112-restart1`](runs/a10-20261112-restart1/HANDOFF.md), three seeds verified TRAINING on 12 November 2026. Use its `ops/status.py` and existing sync services. [Published log snapshot](runs/a10-20261112-restart1/publication-snapshot-20261112/) is dated and contains no checkpoints. The original non-restart attempt described below was stopped and is now in the ignored local archive. No final evaluation is available for the restart snapshot.

# E05 — joint Anderson tolerance 1e-5, cap 10

Explicitly authorized 12 November 2026, including concurrent placement with E04.
Three fresh seeds1/2/3, MultiGoalDockTight-v0, 200000 real transitions, strict FP32,
unchanged E01/E02 scientific configuration and E02 RTX5090 runtime.
Only the requested joint Anderson cap changes to10. Native tolerance stays1e-5.

GPU0/1/2 each host one E05 seed and the matching identified E04 cap100 seed.
Unknown processes block admission. No retries, resume, deletion or raw archive.
Checkpoint interval25000. Final stochastic evaluation1000 episodes, seed26152.

Shared evaluation ordering: E05 finishes training, then waits for E04 evaluation.
E04 remains unchanged. Its separate sync publisher verifies every checkpoint
but delays final-checkpoint acknowledgment until the corresponding E05 training
has a real exit receipt. This prevents E04's existing exclusive-evaluation
admission check from failing because of an E05 co-runner. E05 evaluation begins
only after E04 has terminated and the GPU is free.

Shared-GPU intervals are recorded. Training SPS is observational and must not be
pooled with exclusive-GPU E02 measurements as a controlled solver speed ratio.

Commands, frozen hashes and placement contract: runs/a10-20261112/ops/manifest.json.
Status: runs/a10-20261112/ops/monitor_state.json and monitor_history.jsonl.
Full raw stays on the server. Local mirror includes all checkpoints, events,
compact metadata, actual exits and evaluation records, with SHA256 verification.
Status: preparation and preflight. No production seed submitted yet.
