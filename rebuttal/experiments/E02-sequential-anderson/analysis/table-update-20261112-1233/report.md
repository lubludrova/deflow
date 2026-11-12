# Current quality and timing tables

Observed 2026-11-12T12:36:21.406234+08:00. Evaluation JSON hashes match accepted receipts and real exit 0 markers. Each quality result uses 1000 stochastic evaluation episodes after 200k transitions. The Newton s3 FAILED labels in the older coordinator summaries are administrative admission failures, superseded for evaluation by its explicit ACCEPTED/EXIT receipts. No scientific retraining was inferred.

| Method | Evaluated/planned seeds | U4 | Success (%) |
|---|---:|---:|---:|
| Joint Anderson (A25) | 3/3 | 0.6652 ± 0.0033 | 96.17 ± 1.10 |
| Sequential damped Newton (seed 3 only) | 1/3 | 0.4452 | 86.40 |
| Sequential Anderson | 0/3 | Pending | Pending |

Mean±sample SD across seeds for joint Anderson. Newton s1 is running, s2 failed at 131908 with nonfinite actor loss. Sequential Anderson trainings are at approximately 106k–109k in the 12:33 snapshot, not final evaluations. Intermediate training success must not populate this table.

| Method | A2-trained input: median ms | Relative to joint | N3-trained input: median ms | Relative to joint |
|---|---:|---:|---:|---:|
| Joint Anderson | 19.55 | 1.00× | 19.81 | 1.00× |
| Sequential Anderson | 66.82 | 3.42× | 66.63 | 3.36× |
| Sequential damped Newton | 274.21 | 14.03× | 1197.80 | 60.46× |

Identical saved actor weights, observations and Gaussian noise within each batch, B=1024, T=4, d=2, same RTX 5090, strict FP32. A2 at 99336 and N3 at 78398. One warmup plus three synchronized repetitions, median reported. No co-resident GPU compute, but concurrent preflight on GPU 1 means host not isolated. All rows retained.

Fresh residual failures out of 1024: joint 363/841, sequential Anderson 287/838, Newton 4/556 for A2/N3 respectively. These are not matched-accuracy benchmarks. The supplied three implementations are compared at their recorded production presets. Finite trajectories are not necessarily converged roots.

Safe claim: On these two frozen-input batches, sequential Anderson required 3.36–3.42× the forward-solve latency of joint Anderson. Sequential damped Newton required 14.03–60.46×. This is preliminary evidence for the latency of the tested implementations, not a universal complexity claim or a full-training speedup.

Observed training expense: Newton s3=19.7947h, joint Anderson s3=2.8326h, ratio 6.9882×. Both completed 200k transitions. Shared-GPU placement and changing co-runners make this observational campaign cost, not a controlled solver efficiency ratio. Sequential Anderson training is unfinished and uses different hardware/runtime from E01, so no final end-to-end ratio is available.

For a stronger final claim, use more synchronized repetitions and several preregistered early/mid/late frozen checkpoints, report achieved residual/error beside latency, and measure forward separately from attachment/density/backward. Run only after available GPU capacity is confirmed. No new GPU benchmarks were started for this table update.
