# E03 — tighter Anderson on the E01 paired audit inputs

User-authorized on 12 November 2026 to fill the missing numerical measurements
for Anderson tolerance 1e-6, cap 100 in the three-setting dkt6 response.
Related concerns: D-C10/D-C11/D-C13. No training, resume, or scientific retries.

Completed: 9/9 audits exited successfully and all raw outputs were copied with
SHA-256 verification. All reference tensors, masks and valid reference readouts
match E01 exactly. Five batches are fully valid, with 5,120 trajectories in the
error aggregate. Four batches retain five invalid reference trajectories.
The tighter-setting row is 74.13 ms, 2.70e-6 P95 absolute log-density error and
3.89e-6 median absolute actor-loss error. See [REPORT.md](analysis/REPORT.md),
[metrics.json](analysis/metrics.json) and the
[completion receipt](runs/tol1e-6-cap100-20261112/COMPLETED.json).
All task-owned audit processes exited and GPU 1 was free at final verification.

## Frozen comparison

Use precisely the nine E01 A25 actor-gradient capsules (seeds 1/2/3 near
25k/100k/200k), with unchanged weights, observations, noise, critics, temperature,
effective Jacobian penalty, density safeguards and backward implementation.
This measures the tighter **solver setting on A25-trained policies**, not
numerical accuracy of independently trained T policies. Success and return
remain separate measurements from the three completed T training runs.

The unchanged E01 audit command supports the missing tolerance/cap directly.
Its built-in Newton comparison is also retained as an implementation check,
not introduced as another method in the requested table. E01 source and results
are not modified. Exact commands, capsule hashes, source anchor, resource and
retention policy are recorded in `runs/tol1e-6-cap100-20261112/manifest.json`.

## Measurement and acceptance

- Same server, strict FP32 runtime and exclusive physical RTX 4090 GPU 1 as E01.
- Two warmups, ten synchronized solver-only repetitions per batch of 1,024.
  Includes diagnostics, excludes density/readout/backward and training logging.
- Independent FP64 equilibrium reference with unchanged validity checks.
  Compare the saved observations, noise, reference roots and validity masks
  against E01. Retain every invalid case rather than repairing or filtering it.
- Timing aggregates nine case medians. Actor-loss error is the median absolute
  full-batch loss difference over fully valid references. Log-density P95 pools
  trajectories only from those same fully valid batches. Preserve gradient
  comparisons in raw evidence, but omit them from the requested table.
- Require exit zero, complete raw/summary files, source/input hashes, and a
  verified local copy before proceeding to the next case. No automatic retry.
- Preserve all inputs, outputs, logs and failed attempts. No remote deletion.

Only this new experiment directory is authored. The reviewer response remains
unchanged. Final measured results and their denominators belong in `analysis/`.
