# Numerical preflight

`numerical_preflight.py` does frozen solves only. It never starts training or changes scientific solver tolerances, iterations, damping, backward or density formulas.

```
python validation/numerical_preflight.py --root /absolute/revision3 \
  --output /absolute/new-fixed-preflight --phase fixed --device cuda
python validation/numerical_preflight.py --root /absolute/revision3 \
  --output /absolute/new-fresh-preflight --phase fresh \
  --fresh-bank /absolute/fresh-bank.json --device cuda
```

Optional `--engine` overrides ROOT/DEFlow/src/docking_full/sac_anderson_flow.py. `--fixed-bank` can relocate paths but must preserve all three hashes in `historical_fixed_bank.json`. `--phase all` runs both banks. Each output directory must be new. Exit 0 and `verdict.json["pass"] == true` mean implementation gates pass, **not universal equation convergence**.

A bank is a JSON array with records `path`, `sha256`, `actual_step`, `source`, `rule`. Fresh bank source names must be exactly `newton`, `anderson`, `anderson_highcap`, `anderson_tighter`. Each record selects the latest eligible full1024 training/actor_grad capsule from its pilot. The script verifies hashes, metadata, required actor/critic/loss contents, strict policy and fresh source provenance, and checks no later eligible capsule was omitted. Historical TF32-trained capsules are explicitly permitted as frozen numerical fixtures.

Every case stores `forward_raw.pt` before expensive downstream checks, final `raw.pt`, and `case.json`. Raw data preserve all rows and local steps, strict masks, fixed-q residuals in flattened/per-step/chunks1/32/1024/reversed order, fixed-q FP64 residuals, full FP64 solves, independent full-bank SciPy reference, three-step-size all-row Jacobian finite differences, hard-row FP64 traces, and full1024 native-versus-audit SAC gradient/density parity. No implicit-gradient equality to finite-program finite differences is required. Full-batch FP64 reference gradients are **not computed by this preflight**. Reference validity is recorded, and invalid rows make such a gradient reference unavailable. The later full audit computes it only when every row validates.

All gates and thresholds are fixed in `THRESHOLDS` before execution. Exact FP32 native and fresh residual masks retain rtol1e-5/atol1e-7. An explained boundary is a separate label, not a changed success mask. Its uncertainty is measured fixed-q norm disagreement against FP64 plus `8*eps32*max(residual norms)` norm-rounding allowance and the actual FP32/FP64 tolerance difference. The reference threshold margin must fit this envelope and the envelope must be <=10% of the original threshold. This deliberately avoids a blanket state-scale epsilon. Any unexplained native-success discrepancy blocks. A strict-only or FP64-only hard failure blocks for diagnosis. Only the preregistered exact historical A2 SHA row473/step2 shared failure can pass implementation gates while remaining a scientific failure. Any other shared failure blocks for numerical review, even if FP64 reproduces it. The known A2 row473/step2 must retain status4 in both solves and an invalid independent reference, otherwise it is flagged for review.

FD steps are [1e-5,1e-6,1e-7] times max(1,absolute coordinate), with every local Jacobian requiring one step with absolute Frobenius error<=1e-7 or relative<=1e-6. Readout parity uses atol1e-6/rtol1e-4 and normalized gradient difference<=1e-3. Finite outputs, gradients and residual quantities are mandatory. Historical single-row traces are independent diagnostics, not a claim that the production vectorized solver emitted an internal trace.

Run bounded local regression tests with:

```
python -m unittest discover -s validation -p 'test_*.py' -v
```

`run_synthetic_cpu_check.py` makes a new `synthetic_cpu_*` directory and runs the complete case pipeline with a small-width neural field and all1024 synthetic rows. These artifacts are tests only and must not enter scientific results. Seven regression tests and the synthetic full pipeline passed locally with Torch2.5.1 CPU. Production Torch2.5.1 CUDA gates remain for the deployment runner.
