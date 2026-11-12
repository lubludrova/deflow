# Completed tighter-Anderson audit

| Anderson tolerance / cap | Forward ms/batch | P95 absolute log-density error (nats) | Median absolute actor-loss error | Success (%) | Return |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1e-5 / 25 | 21.83 | 7.54e-3 | 3.66e-4 | 96.17 ± 1.10 | -7.87 ± 0.43 |
| 1e-5 / 500 | 53.34 | 4.65e-6 | 5.19e-5 | 98.20 ± 0.70 | -7.77 ± 0.41 |
| 1e-6 / 100 | 74.13 | 2.70e-6 | 3.89e-6 | 96.53 ± 1.60 | -7.62 ± 0.35 |

Timing and errors compare solver settings on the same nine saved A25-policy batches.
Success and return are separate training results, mean ± sample SD over three seeds,
with 1,000 stochastic episodes per policy using its training solver.
Timing: RTX 4090, median of nine case medians, each with ten repetitions after two warmups.
Includes solver diagnostics, excludes density/readout/backward and training telemetry.
Errors: five fully valid FP64-reference batches. Log-density P95 pools 5,120 trajectories.
Five reference-invalid trajectories affect four batches: all three near 100k and seed 3 near 200k.
Those full-batch errors remain unavailable. All reference tensors and valid reference readouts
match E01 exactly. New timing was measured in a separate exclusive session on the same GPU.
Small scalar loss error does not certify gradient accuracy. Safeguarded equilibrium agreement
does not certify exact finite-iteration sampler densities or sensitivities.
