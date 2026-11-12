# Validation — 12 November 2026

27 unit/integration tests passed. Native engine source unchanged. GPU smoke training reached 256 transitions and its checkpoint passed a 10-episode evaluation. Production configuration dictionaries matched E01 except seed/run names. The 8000-transition production-batch pilot is recorded separately.

## Frozen capsules

Both inputs use all 1024 actor-gradient rows. A2 is the saved capsule nearest 100k (99336), N3 is the previously selected late diagnostic capsule at 78398. Selection and hashes are in `runs/sa-20261112/validation/inputs.json`. No failed row was dropped.

| Capsule | Method | Fresh failed samples / 1024 | Median forward ms |
|---|---|---:|---:|
| a2_step99336.pt | anderson | 363 | 19.55 |
| a2_step99336.pt | newton | 4 | 274.21 |
| a2_step99336.pt | sequential_anderson | 287 | 66.82 |
| n3_step78398.pt | anderson | 841 | 19.81 |
| n3_step78398.pt | newton | 556 | 1197.80 |
| n3_step78398.pt | sequential_anderson | 838 | 66.63 |

GPU0 had no co-resident compute. Three synchronized samples followed one warmup. GPU1 concurrently performed preflight work, so the host was not isolated. These are small fixed-capsule measurements, not training-cost ratios, matched-accuracy timings, or a general speed ranking.

All trajectories, density values, SAC losses and gradients were finite. The FP64 sequential Newton diagnostic reference passed fresh residual and nonsingularity checks for 1022/1024 A2 rows and 468/1024 N3 rows. This does not prove branch uniqueness. Error summaries use valid rows. Full-batch gradient differences explicitly include reference-invalid rows and must not be interpreted as certified gradient errors. Raw tensors, exact validity masks and failed rows are retained in the `_raw.pt` artifacts.

Sequential Anderson improves some residual statistics versus joint Anderson at nominal cap25, but many samples still fail the common test. The difficult N3-trained policy remains difficult. Passing implementation gates means the code and gradient attachment are functional, not that every root converges or that the baseline will learn well.

## Verification corrections

The first test pass exposed overly strict FP32 equality requirements. The one-step parity test now allows 1e-6 relative / 1e-7 absolute rounding error. The synthetic failure-propagation test uses FP64 to test its intended 1e-10 threshold. Scientific solver tolerances were not changed.

The initial frozen-capsule diagnostic had two integration issues: the inherited critic helper expected concatenated inputs, and importing the parent actor module changes a global engine constructor. The diagnostic now adapts the critic input and uses the saved native constructor. Failed diagnostic outputs/logs remain under `fixed-capsules*`. Final evidence is `fixed-capsules-v3/report.json`. The initial smoke evaluation referenced an incorrect checkpoint filename and was rerun against the checkpoint listed in DONE.json.

## Interpretation

The requested baseline is implemented as a forward-solver ablation with common density/backward. It does not establish that sequential solving is inherently better or worse. Matching seeds and interaction budget does not match computational work, hardware or runtime. New runs use RTX5090 / torch2.7.1+cu128. E01 used torch2.5.1+cu124 and had shared-GPU periods.

Production-batch pilot completed at 8002 transitions in 341.18 seconds. Actor, saved critics and temperature are finite, all logged actor/critic losses are finite, strict FP32 was recorded through backward. Four supervisor tests passed, including changed-source rejection, duplicate-attempt refusal and preservation of a real nonzero child exit. All three RTX5090 devices passed a CUDA forward/backward check. RELEASE_READY.json records the gate. Target critics are not part of the inherited lightweight checkpoint, so release validation checks the saved actor/q1/q2/log_alpha components.

Post-launch operational observation: seed2 initial process identity captured an empty /proc cmdline. Stable PID/start ticks/boot ID and exact manifest argv were independently verified. A separate read-only status overlay and supplemental receipt resolve the display without changing frozen sources or restarting any training. See runs/sa-20261112/ops/identity_reconciliation_s2.json.
