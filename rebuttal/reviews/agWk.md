# Response to agWk — working draft

**Draft, not submitted.** Based on the original paper and 8 November notes. E01 results are not incorporated yet. Validate all numbers before submission.

[Concern checklist](../common/ISSUES.md#reviewer-agwk) · [Shared drafting notes](../common/response-notes.md)

**A1 — When the cost may be worthwhile.** Precision-constrained successful diversity is the clearest supported use case. In docking, reaching one preferred goal is insufficient to maximize successful coverage. Table 1's mean U4 advantage appears in all four variants, with the largest descriptive relative gains in the tight variants, although their seed uncertainty is substantial. This makes implicit sampling a candidate when preserving multiple narrow successful action modes has value and additional computation is affordable. It is not a validated universal task-selection rule. For tasks already served well by Gaussian or explicit-flow policies, or when elapsed time is the binding budget, the current results do not establish a reason to prefer DEFlow. MuJoCo and mixed final manipulation success illustrate that boundary.

**A2 — A shorter theory explanation.** The construction has three parts. First, implicit Euler evaluates the field at the unknown next action, so sampling requires solving an equation. Second, at a differentiable equilibrium with an invertible system Jacobian, implicit differentiation obtains policy sensitivities by a linear solve, without differentiating through the root-finder iterations. Third, each exact implicit step has Jacobian $(I-hJ_k)^{-1}$, where $J_k$ is the field Jacobian. Change of variables therefore adds $\log|\det(I-hJ_k)|$ to the log-density, followed by the final tanh correction. Appendix A states the selected-branch and regularity assumptions. The scalar contraction example motivates stability under few steps. It does not guarantee convergence of a nonlinear solver or accuracy of the finite implementation.

