Thank you for highlighting both the technical contribution of combining implicit flow policies with maximum-entropy learning and the evidence for successful diversity under precision constraints. Your request for a sequential implicit baseline identifies an important attribution gap. We address that comparison first, then numerical reliability during learning, and finally the practical trade-off.

**W1: “Insufficient ablation to isolate the contribution of the joint deep-equilibrium formulation from implicit Euler integration.”**

Equations (7) and (8) clarify this distinction: summing the sequential implicit-Euler updates gives the joint equations, while subtracting consecutive joint equations recovers the sequential updates. They therefore describe the same exact trajectories and endpoint policy when the same solution branch is selected. Our choice of a joint formulation with Anderson acceleration was motivated by its computational structure: each solver iteration evaluates the velocity field across all trajectory points in a batch and updates the entire trajectory together, without requiring each implicit step to converge before proceeding to the next. This provides an opportunity to exploit parallel computation. 

To address the gap you identified, we have extended the experiments to compare:

- **Joint Anderson (DEFlow baseline):** The original DEFlow solver, which applies Anderson acceleration to the joint fixed-point system over the entire implicit-Euler trajectory.
- **Sequential Anderson:** A step-by-step solver that applies the same Anderson acceleration algorithm to the local fixed-point equation at each implicit-Euler step.

The comparison uses three training seeds on wide-navigation/tight-docking MultiGoal, with 200k transitions and 1,000 stochastic evaluation episodes.

| Method |Seeds | U4 ↑ | Success (%) ↑ | A2 forward solve, ms ↓ | N3 forward solve, ms↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Joint Anderson (DEFlow baseline) | 3 | 0.6652 ± 0.0033 | 96.17 ± 1.10 | 19.55 (1.00×) | 19.81 (1.00×) |
| Sequential Anderson | 3 | 0.6197 ± 0.0568 | 92.13 ± 5.41 | 66.82 (3.42×) | 66.63 (3.36×) |

For both solvers, we measure the success and U4 (Eq. 17) forward-solve latency using the same fixed model weights and inputs on the same GPU, reporting the median of three synchronized runs after warmup. Joint Anderson achieved higher mean policy performance and lower observed variability, while sequential Anderson required **3.36–3.42× the forward-solve time**. Overall, these results support our choice of joint Anderson as a practical solver for DEFlow in the tested setting. Compared with sequential Anderson, it achieved higher mean goal coverage and success, with lower variation across the three training seeds, while requiring approximately 3.4× less forward-solve time and can train ~3.5x times faster. Comparing joint and sequential Anderson helps isolate the effect of solving the trajectory jointly, and the observed results support this design choice in DEFlow.

**W2: “Limited practical performance improvements relative to the computational and numerical costs.”**

We agree that DEFlow's additional computational cost must be justified by a concrete benefit. Our results support two such benefits:

1. **Higher successful coverage under precision constraints.** We claim that DEFlow has higher learning ability by achieving the highest mean successful coverage on all four MultiGoal docking variants (Table 1). For example, on wide-navigation/loose-docking, its U4 is 0.671 ± 0.002, compared with 0.634 ± 0.021 for the strongest baseline, DIME. Across the four variants, the mean relative improvement over DIME is approximately 10.1%. We believe the main value lies in achieving higher successful coverage on challenging precision-constrained tasks, where stiff learned flow dynamics can make accurate sampling with few integration steps difficult.

2. **Higher sample efficiency.** Across the five MuJoCo and five manipulation tasks, DEFlow achieves higher mean return AUC than SAC-Flow on 8/10 tasks and both SAC and DIME on 7/10 tasks each, suggesting improved sample efficiency under the evaluated training recipes (Tables 7–8). Additionally, at 100k training steps, DEFlow achieves returns of 25.7 ± 6.8 on PickCube and 40.5 ± 0.4 on PushCube, compared with 18.3 ± 1.8 and 39.4 ± 1.0 for SAC, 17.2 ± 3.1 and 31.9 ± 1.7 for SAC-Flow, and 20.2 ± 1.8 and 32.6 ± 8.3 for DIME, respectively. Averaging the DEFlow-to-baseline return ratios across both tasks and all three baselines yields a mean relative improvement of **28.5%** at this checkpoint. Thus, we are simply more sample efficient and we believe that in some cases where environment is too expensive this benefit can make the learning more efficient.

**Q: “How reliable are the policy gradients and action densities when the equilibrium solver does not converge?”**

Thank you for raising this concern. We agree that our original final-checkpoint analysis was insufficient to establish reliability during learning. Without convergence, both quantities are approximations whose accuracy requires checking. To answer this concern, we decided to conduct one more ablation study, trying to show how different the DEFlow performs on docking task with different tolerance and cap settings.
| Anderson setting (tolerance / cap) | Forward time (ms/batch) | Log-density (nat), Median | Log-density (nat), P95 | Gradient (%), Median | Gradient (%), P95 | Success (%) |
|---|---:|---:|---:|---:|---:|---:|
| 1e-5 / 10 | 7.63 | 1.47e-4 | 9.19e-2 | 207 | 315 | 50.63 ± 6.86 |
| 1e-5 / 25 | 18.76 | 7.77e-7 | 7.54e-3 | 8.02 | 59.0 | 96.17 ± 1.10 |
| 1e-5 / 500 | 44.73 | 4.10e-7 | 4.39e-6 | 0.395 | 2.57 | 98.20 ± 0.70 |
| 1e-5 / 100 | 45.23 | 4.09e-7 | 4.97e-6 | 0.395 | 2.57 | 93.40 ± 6.93 |
| 1e-6 / 100 | 64.22 | 3.95e-7 | 2.70e-6 | 0.0343 | 0.0946 | 96.53 ± 1.60 |

All reported solver comparisons use the same Jacobian-regularization schedule: β increases linearly from zero to 0.01 over the first 100k transitions, with a Jacobian-norm threshold of 0.9. We have not conducted a separate β ablation, so these experiments assess sensitivity to solver settings while leaving sensitivity to the regularization strength unresolved.
