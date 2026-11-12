# Paired solver audits

| Solver | Median forward ms | P95 logp error | Median relative gradient error | Median absolute delta actor loss | Timing / gradient cases | Logp trajectories |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| anderson_cap25 | 21.8264 | 0.00754202 | 0.0801736 | 0.000366171 | 9 / 5 | 5120 |
| anderson_cap100 | 53.2525 | 4.95673e-06 | 0.0039502 | 5.18958e-05 | 9 / 5 | 5120 |
| anderson_cap500 | 53.3401 | 4.64584e-06 | 0.0039502 | 5.18958e-05 | 9 / 5 | 5120 |
| newton | 207.881 | 7.90327e-06 | 7.5525e-05 | 2.88115e-06 | 9 / 5 | 5120 |

| Audit | State | Target / actual step | Valid / invalid reference rows | Full reference usable |
| --- | --- | --- | --- | --- |
| audit_a1_25000 | FINISHED | 25000 / 26726 | 1024 / 0 | True |
| audit_a2_25000 | FINISHED | 25000 / 24102 | 1024 / 0 | True |
| audit_a3_25000 | FINISHED | 25000 / 24087 | 1024 / 0 | True |
| audit_a1_100000 | FINISHED | 100000 / 100242 | 1022 / 2 | False |
| audit_a2_100000 | FINISHED | 100000 / 99336 | 1023 / 1 | False |
| audit_a3_100000 | FINISHED | 100000 / 100163 | 1023 / 1 | False |
| audit_a1_200000 | FINISHED | 200000 / 195671 | 1024 / 0 | True |
| audit_a2_200000 | FINISHED | 200000 / 195664 | 1024 / 0 | True |
| audit_a3_200000 | FINISHED | 200000 / 197397 | 1023 / 1 | False |

Median across finished audit cases for timing/gradient/absolute loss error. Logp P95 pools trajectories only from usable complete-batch references. Explicit denominators accompany every solver.

Invalid references and failed candidates remain in per-audit rows and raw.pt.
No complete-batch gradient comparison when any reference row is invalid.
A-policy capsule evidence does not cover all N-policy failure regimes.
The safeguarded equilibrium reference does not certify finite-sampler density.
A100 here has tol1e-5 and is distinct from the T training arm at tol1e-6.

Signed per-case loss deltas, residual tails and all case denominators are in metrics.json and per_audit.csv.
