# DEFlow solver campaign

Training and evaluation are separate. All seeds, failures and deferred jobs are retained.

| Run | State | U4 | Success | Return | Train seconds | Eval seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| newton_s1 | INTERRUPTED |  |  |  |  |  |
| newton_s2 | FAILED |  |  |  |  |  |
| newton_s3 | FAILED |  |  |  | 71264.60828971863 |  |
| anderson_s1 | BLOCKED |  |  |  |  |  |
| anderson_s2 | EVALUATION_FINISHED | 0.66145462359175 | 0.949 | -8.105148948183865 | 8571.401953458786 | 6.227883577346802 |
| anderson_s3 | BLOCKED |  |  |  |  |  |
| anderson_tighter_s1 | BLOCKED |  |  |  |  |  |
| anderson_tighter_s2 | BLOCKED |  |  |  |  |  |
| anderson_tighter_s3 | BLOCKED |  |  |  |  |  |
| anderson_highcap_s1 | BLOCKED |  |  |  |  |  |
| anderson_highcap_s2 | BLOCKED |  |  |  |  |  |
| anderson_highcap_s3 | BLOCKED |  |  |  |  |  |

## Frozen paired audit

| Audit | State | Target | Actual | Valid FP64 | Invalid FP64 |
| --- | --- | ---: | ---: | ---: | ---: |
| audit_a1_25000 | BLOCKED | 25000 |  |  |  |
| audit_a1_100000 | BLOCKED | 100000 |  |  |  |
| audit_a1_200000 | BLOCKED | 200000 |  |  |  |
| audit_a2_25000 | BLOCKED | 25000 |  |  |  |
| audit_a2_100000 | BLOCKED | 100000 |  |  |  |
| audit_a2_200000 | BLOCKED | 200000 |  |  |  |
| audit_a3_25000 | BLOCKED | 25000 |  |  |  |
| audit_a3_100000 | BLOCKED | 100000 |  |  |  |
| audit_a3_200000 | BLOCKED | 200000 |  |  |  |

Full solver curves, per-example tails, paired actions/logp/SAC gradients and reference exclusions:
`audits/*/summary.json`, `audits/*/raw.pt`. Per-role training/evaluation denominators:
`campaign_report.json` and each `telemetry_summary.json`. Raw all-call telemetry stays on the server.

A/H changes the iteration budget. A/T changes both budget and tolerance. Cap-reaching and convergence failure are separate.
Cap500 is not a convergence certificate. This is neither exact finite-sampler likelihood certification nor an equal-wall-clock SAC/DIME comparison.
