# Command reference

Scientific queue is defined by `night_campaign.json`. The three-Newton commands below illustrate the original arm only. For the other arms use each job’s exact argv in the night manifest.

## Command templates

Run from the directory containing the `deflow_rebuttal` package, with the tested environment's Python. Replace the paths with the supplied server locations. Each output directory must be new.

Read-only plan for one seed:

```bash
python -m deflow_rebuttal.run_docking --repo /path/to/DEFlow --solver newton --seed 1 --steps 200000 --output /absolute/results/newton_wide_tight_s1 --plan
```

Bounded integration smoke, separate from scientific outputs:

```bash
python -m deflow_rebuttal.run_docking --repo /path/to/DEFlow --solver newton --seed 0 --output /absolute/results/smoke_newton --smoke --execute
python -m deflow_rebuttal.run_docking --repo /path/to/DEFlow --solver anderson --seed 0 --output /absolute/results/smoke_anderson --smoke --execute
```

Formal execution, after preflight:

```bash
python -m deflow_rebuttal.run_docking --repo /path/to/DEFlow --solver newton --seed 1 --steps 200000 --output /absolute/results/newton_wide_tight_s1 --execute
python -m deflow_rebuttal.run_docking --repo /path/to/DEFlow --solver newton --seed 2 --steps 200000 --output /absolute/results/newton_wide_tight_s2 --execute
python -m deflow_rebuttal.run_docking --repo /path/to/DEFlow --solver newton --seed 3 --steps 200000 --output /absolute/results/newton_wide_tight_s3 --execute
```

Schedule according to confirmed resources. Use one trainer per explicitly assigned GPU, following night_campaign.json. An absent server assignment, unavailable original model/source identity, failed test, unreliable reference or missing required telemetry must be surfaced before the corresponding dependent experiment.

## Historical frozen-bank preparation and audit

The following example uses the actual relative path of wide-tight seed1's final checkpoint. Repeat for the preselected early/mid/final checkpoints and seeds1/2/3 after inventorying their exact names. First run a small technical pilot to estimate reference runtime. The full batch is1024. Do not replace it with the smoke batch when reporting production-batch behavior.

```bash
python -m deflow_rebuttal.prepare_inputs --repo /path/to/DEFlow --checkpoint /path/to/DEFlow/results/docking/wide_tight/deflow/seed_01/checkpoints/ckpt_step_000200005.pt --output /absolute/audit_inputs/wide_tight_s1_final.pt --episodes 200 --batch-size 1024 --seed 26152 --device cuda
python -m deflow_rebuttal.audit --engine /path/to/DEFlow/src/docking_full/sac_anderson_flow.py --checkpoint /path/to/DEFlow/results/docking/wide_tight/deflow/seed_01/checkpoints/ckpt_step_000200005.pt --inputs /absolute/audit_inputs/wide_tight_s1_final.pt --output /absolute/audits/wide_tight_s1_final --device cuda --newton-tol 1e-5 --newton-cap 25 --warmup 2 --repeats 10 --probe-loss
```

`prepare_inputs` gathers states visited by the historical policy, includes unsuccessful episodes, fixes noise and records hashes/selected indexes. This is a **reconstructed observation/noise batch, not historical replay or an original training minibatch**. If too few states were gathered, increase episodes and retain the fixed seed. It intentionally refuses to silently duplicate states.

The historical command explicitly uses `--probe-loss`. Its gradient is for the labeled action/logp probe, **not the SAC objective**. A historical SAC audit needs separately verified alpha, effective Jacobian penalty and clipping settings passed using `--loss-config`. Do not infer an original target-critic Bellman objective from online critics alone.

For a newly captured `training/actor_grad` capsule containing actor, q1/q2, alpha and `loss_config`, use that **same capsule file** as both `--checkpoint` and `--inputs`. Replace the example filename with the actual selected capsule whose metadata confirms the role and full batch. Auto-detected loss metadata then enables the configured SAC actor-loss audit:

```bash
python -m deflow_rebuttal.audit --engine /path/to/DEFlow/src/docking_full/sac_anderson_flow.py --checkpoint /absolute/results/newton_wide_tight_s1/capsule_00001000.pt --inputs /absolute/results/newton_wide_tight_s1/capsule_00001000.pt --output /absolute/audits/newton_s1_actor_capsule --device cuda --warmup 2 --repeats 10 --fd-step 0.001
```

Capsule1000 is a filename template, not a promise that this call is an actor update. Inspect metadata and select the appropriate saved actor capsule. `--fd-step` provides one directional finite-program diagnostic, not a global-gradient or likelihood certification. Repeat at other scales separately if interpreting that diagnostic. The reference is a validated converged **safeguarded** readout with the common shifted adjoint. Solver timing repetitions exclude readout/telemetry and are not independent training seeds.

## Final solver-aware stochastic evaluation

After each run, read its `DONE.json` and choose the checkpoint listed there at the required horizon. For the ordinary200k save convention the following is the expected path. Verify the actual entry rather than guessing if names differ.

```bash
python -m deflow_rebuttal.evaluate --repo /path/to/DEFlow --run /absolute/results/newton_wide_tight_s1 --checkpoint /absolute/results/newton_wide_tight_s1/runs/newton_wide_tight_s1/ACTUAL_CHECKPOINT_FROM_DONE_JSON.pt --output /absolute/evaluations/newton_wide_tight_s1_final --episodes 1000 --eval-seed 26152 --device cuda
```

Repeat for seeds2/3 with their own run/checkpoint/output paths and the same predetermined evaluation seed/protocol. This command reads the run manifest, validates source and solver/seed identity, reconstructs the recorded solver and writes `evaluation.json` plus numerical telemetry. Training `DONE.json` by itself does not mean this final evaluation has completed. Preserve and synchronize both artifact groups.

Checkpoint paths above are placeholders: read DONE.json checkpoints[].path because real transitions can overshoot the requested horizon by up to7. Never assume exact filename200000.

## Anderson convergence budget

High-cap training uses the unchanged native Anderson implementation and **the same tolerance** as the original arm. There is no unbounded mode. The safety cap is 500, with native batch-mean early stopping. Record cap-reaching separately from failure to meet tolerance at that cap.

```bash
python -m deflow_rebuttal.run_docking --repo /path/to/DEFlow --solver anderson --anderson-tol 1e-5 --anderson-cap 500 --seed 1 --steps 200000 --output /absolute/results/anderson_highcap_s1 --execute
```

Paired frozen audit, one independent FP64 reference per capsule, common obs/noise/parameters/critics/alpha and readout. Choose the same full-batch training/actor_grad capsule as checkpoint and inputs. Preselect nearest available capsules to 25k/100k/200k for each seed and report actual steps. Do not select only failures. This command also retains the Newton comparison:

```bash
python -m deflow_rebuttal.audit --engine /path/to/DEFlow/src/docking_full/sac_anderson_flow.py --checkpoint /absolute/results/anderson_s1/SELECTED_ACTOR_GRAD_CAPSULE.pt --inputs /absolute/results/anderson_s1/SELECTED_ACTOR_GRAD_CAPSULE.pt --anderson-caps 25 100 500 --anderson-tol 1e-5 --output /absolute/audits/anderson_s1_budget_sweep --device cuda --warmup 2 --repeats 10
```

`summary.json` includes each cap's native residual curve, stopping reason, iteration count, per-sample tail statistics, paired action/logp/gradient changes against the highest cap, and independent reference comparisons. `raw.pt` preserves per-example trajectories/residuals/readouts. Native curve element zero corresponds to iteration count 3. Do not average iteration counts only over successful calls: capped failures are censored, not observed convergence times. Repeated solves can differ slightly due to numerical linear algebra. The highest cap is a comparator, not a guaranteed root. Nonfinite data remain in raw artifacts.

Training JSONL records `mean_criterion_met`, `iteration_limit_reached`, and `budget_exhausted_without_convergence`. At the last allowed iteration both limit-reached and convergence can be true. Native batch-mean convergence and the common per-step residual test are different criteria. Retain both. The safety cap stays fixed throughout the campaign. If many calls reach 500 without convergence, report that finding rather than raising the cap silently.
