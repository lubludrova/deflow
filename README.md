# DEFlow: Deep Equilibrium Flow Maximum-Entropy Reinforcement Learning

## Abstract

Expressive flow policies can represent multiple high-value actions, but sampling them with few integration steps while retaining tractable action densities remains challenging. We introduce DEFlow, a deep equilibrium flow policy that replaces explicit sampling with implicit integration and solves all intermediate actions jointly as an equilibrium system. We combine implicit differentiation with a discrete change-of-variables formula for the final-action density, enabling off-policy training within soft actor–critic without backpropagating through solver iterations. The approach targets tasks requiring both diversity and precision, at the cost of additional nonlinear and linear solves.

## Rebuild figures and tables from published results

Install the analysis dependencies with `python -m pip install -r requirements.txt`.
Run the following commands from the repository root. They read the data in
`results/` directly; no private documentation, selection manifests, model weights,
or simulators are needed. Outputs are written to the ignored `figures/` directory.
The manipulation figure command requires a new output directory.
The docking task schematic is an illustration of the environment geometry defined
in the plotting code, rather than an empirical result; it is retained separately
from the evaluation curves.

```bash
python src/analysis/continuous_control_preview.py --results-root results/mujoco --output-dir figures/mujoco
python src/analysis/continuous_control_sensitivity.py --results-root results/mujoco --output-dir figures/mujoco_sensitivity
python src/analysis/fig_docking_section.py --results-root results/docking --output-dir figures/docking
python src/analysis/fig_manipulation_native.py --results-root results/manipulation --output-dir figures/manipulation --tensorboard-python "$(command -v python)"
python src/analysis/manipulation_run_table.py --input figures/manipulation/data.json --output-dir figures/manipulation
python src/analysis/summarize_manipulation_solver_reeval.py --results-root results/numerical/solver_accuracy --output figures/solver_accuracy/summary.json
python src/analysis/plot_docking_integrator_ablation_minimal.py --shared-inputs results/numerical/integrator_depth/shared_inputs.json --shared-results results/numerical/integrator_depth/shared_results.json --own-results results/numerical/integrator_depth/explicit_own.json results/numerical/integrator_depth/deflow_own.json --bank wide_tight --output-dir figures/integrator_depth
python src/analysis/check_artifact_consistency.py
```

All run directories in each supplied results directory are included. Run identities
come from `<task>/<method>/seed_XX`; solver audits use `<task>/seed_XX`.
The `seed_XX` labels are replicate identifiers, not a specification of the original
training RNG seeds. Do not add unrelated runs to these directories when rebuilding
the published figures.

Docking runs reaching 200k transitions are complete. Three failed runs retain
their recorded failure steps in per-run `status.json` files; their partial curves
are shown separately and do not enter completed-run means. An incomplete docking
run without a status file is rejected. Missing expected input files are errors,
not a reason to silently omit a run.

The checks validate the saved evaluations and derived metrics. They do not verify
the training configurations or checkpoint contents. Checkpoint paths and hashes in
evaluation exports identify the original artifacts; weights are not distributed
with these results. New simulator evaluations require separately supplied weights
and matching runtime inputs.

The published data do not include the Panda illustration inputs (Figure 1),
intermediate stochastic-success evaluations, or training-throughput measurements.
Those exhibits cannot be rebuilt from this results release. The manipulation
learning curves use recorded deterministic evaluations; final sampled-success
tables use the separate episode-level evaluations at 500k steps.
