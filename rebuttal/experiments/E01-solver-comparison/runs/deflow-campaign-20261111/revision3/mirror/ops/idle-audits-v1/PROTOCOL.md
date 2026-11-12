# Paired audit protocol for the dkt6 response

Approved by the user's request to use the idle GPU for remaining experiments,
refined by the supplied response on 12 November 2026. This changes scheduling
only. The frozen audit, actor, reference, density and backward implementations
remain byte-identical. No training retry, resume, seed replacement or new arm.

## Cases and placement

Nine cases: A training seeds 1/2/3 crossed with target transitions
25,000/100,000/200,000. Select the nearest eligible complete 1,024-example
training/actor_grad capsule. Ties choose smaller actual step, then lexical path.
Require saved actor, critics, constructor, loss configuration, observations,
noise and provenance. Preserve exact actual steps and selected file SHA-256.
The same capsule supplies both checkpoint and inputs.

Run serially and exclusively on physical GPU 1. Queue target 25k for seeds
1/2/3, then 100k for seeds 1/2/3, then 200k for seeds 1/2/3. No competing
training or GPU diagnostic may share this device. Other physical GPUs continue
their existing jobs. Keep all original parents until true exits are recorded.
Administrative admission latches prevent superseded controllers from duplicating
audits or evaluations. A new canonical operational report combines all jobs.

## Solvers and objective

Each case evaluates A25/A100/A500 at tolerance 1e-5 and the frozen sequential
Newton at rtol 1e-5, atol 1e-7, cap 25 per Euler step. A100 here is not the T
training arm, whose tolerance is 1e-6. Preserve the native root initialization,
backtracking, readout, density safeguards and shifted dense adjoint.

Use the actual recorded SAC actor loss, including both critics, temperature,
logp clipping bounds and effective Jacobian penalty. No surrogate probe loss.
Candidate precision is strict FP32. Reference roots and reference readout use
the frozen independent FP64 validation, including residual and conditioning
checks. Keep reference-invalid masks and branch discrepancies.

## Measurements and validity

- Forward time: median of ten synchronized solver-only repetitions after two
  warmups, milliseconds per batch of 1,024. Includes solver diagnostics and
  counter extraction, excludes density/readout/training telemetry.
- P95 absolute log-density difference, in nats, against the validated reference.
- L2 relative full-SAC-actor-gradient error and the reference gradient norm.
- Signed actor-loss difference: candidate minus reference. Also retain its
  absolute magnitude to avoid cancellation in aggregate summaries.
- Native convergence, fresh residual tails, attachment changes, and all
  reference validity counts. Call, sample-trajectory and local-step denominators
  are distinct.

Full-batch reference errors are unavailable if any of the 1,024 reference rows
is invalid or the reference readout is nonfinite. Never remove hard rows to
manufacture a full-batch gradient. Retain candidate failures and all raw tensors.
Highest-cap paired discrepancies are secondary comparisons, not validated
reference errors. No reference repair or scientific retry is authorized.

Produce 36 per-case solver rows with explicit validity, plus a four-solver
overview. The overview uses case medians for timing/gradient/absolute loss and a
pooled logp P95 only across complete usable reference cases. Publish all metric
denominators and per-case signed loss deltas. Empty denominators yield N/A.

## Preservation and interpretation

Retain every attempt, log, capsule, summary and raw.pt. Verify SHA-256 and mirror
the nine raw audit outputs as well as selected capsules and final tables.
No remote deletion. Synchronization keeps running through the complete campaign
and stops only after this operational revision's final acknowledgment.

These A-policy cases address D-C02/D-C04 and D-C10/D-C11/D-C13, with relevant
evidence for V-C01/V-C12 and compute concerns. They do not establish the cause
of N2's failure, cover all N-policy regimes, certify finite-sampler density,
or prove that joint solving is mathematically necessary. Separating learning
from final-sampling effects requires a separately specified cross-evaluation.
