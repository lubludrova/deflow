# DEFlow reviews supplied by the author

Source: four screenshots supplied in the conversation, manually transcribed on 2026-11-06. This is a transcription, not an OpenReview API export. The screenshots remain the authority for any transcription discrepancy.

Paper: DEFlow: Deep Equilibrium Flow Maximum-Entropy Reinforcement Learning. ICLR 2027 Conference Submission 26152. Submitted 17 September 2026, modified 6 October 2026. Reviews below display modified 6 November 2026, 20:25.

## Reviewer VrBX

Official review dated 1 November 2026, 08:19.

### Paper Summary

DEFlow replaces explicit flow-policy sampling with implicit Euler integration and jointly solves the intermediate actions as an equilibrium system. It combines implicit differentiation with a final-action change-of-variables density, exact under the stated equilibrium assumptions, to train an expressive policy within SAC. Results support improved successful diversity on docking and earlier reward learning on several manipulation tasks, with matched explicit-flow controls and numerical audits. Benefits come with additional solver cost and approximation error; MuJoCo performance is broadly competitive rather than consistently superior.

Presentation: 3: Clear and easy to follow.

Informativeness: 2: I learned something interesting or useful.

Soundness: 2: I have minor or fixable concerns about the paper's soundness.

### Critical Strengths

1. A coherent way to combine implicit flow sampling with maximum-entropy learning. The same implicit-step structure yields both policy sensitivities and a final-action density expression. This provides a principled alternative to explicit flow rollouts without requiring backpropagation through solver iterations or a path-entropy surrogate.
2. Targeted evidence for successful diversity under precision constraints. The matched explicit-flow control and docking metric test whether policies preserve multiple successful strategies, not merely high return on a preferred route. DEFlow's higher mean coverage across all four variants makes the numerical design empirically relevant, while the disclosed costs and negative results help define its scope.

### Critical Weaknesses

1. The practical density and gradient approximation needs stronger validation during learning. The exact formulas describe a converged implicit solution, whereas training uses finite solves, shifted matrices, and clipping. The numerical audit is useful and reassuring on four tasks, but PushWall's P95 gradient discrepancy remains 50.3% even with tighter evaluation. Comparing against an implicit reference does not fully validate the likelihood of the actual finite-iteration sampler, and unchanged final-policy success does not establish that these approximations are harmless during training.
2. The practical advantage remains limited by compute cost and statistical uncertainty. DEFlow takes about three times as long to train as SAC and has lower throughput than SAC-Flow, but its gains are evaluated at fixed interaction budgets rather than equal elapsed time. Tight-docking improvements and the PushWall return estimate are uncertain across seeds, and the depth-robustness illustration uses one surviving paired seed.

### Questions For Authors

1. How large are the discrepancies between the actual finite-solver sensitivities and the implicit gradients at representative training checkpoints, especially on PushWall? A targeted tighter-solver training comparison would help determine whether the remaining errors change learning, rather than only final evaluation.
2. Do the docking or manipulation advantages persist under equal wall-clock budgets against SAC-Flow and DIME? Please clarify hardware and field-evaluation costs; return-versus-time curves from existing logs, if available, would directly address this trade-off.

Flag For Ethics Review: No.

Rating: 3: Weak acceptance.

Confidence: 1: I do not have sufficient expertise to assess this submission reliably and have alerted the AC.

## Reviewer dkt6

Official review dated 21 October 2026, 15:12.

### Paper Summary

This paper proposes DEFlow, a deep-equilibrium flow policy for maximum-entropy reinforcement learning. The authors identify limitations of explicit Euler integration in flow-based policies, particularly when only a small number of integration steps are available and the target action distribution contains narrow or multiple modes. To address this issue, DEFlow replaces explicit Euler integration with implicit Euler updates and formulates the intermediate action trajectory as a joint deep-equilibrium system. The forward trajectory is obtained through a numerical root solver, while policy gradients are computed using implicit differentiation without backpropagating through the solver iterations. The authors further derive a discrete change-of-variables expression for the final-action log-density, enabling DEFlow to be trained within the Soft Actor-Critic framework. The proposed approach is evaluated on MultiGoal docking, MuJoCo locomotion, and ManiSkill/MetaWorld manipulation tasks. DEFlow achieves the highest mean successful coverage across four MultiGoal variants and the highest mean time-averaged return on four of five manipulation tasks. The paper also includes numerical analyses and comparisons against explicit Euler integration, demonstrating advantages under certain precision-constrained settings. Overall, the paper presents an interesting numerical approach to training expressive flow policies for maximum-entropy reinforcement learning.

Presentation: 3: Clear and easy to follow.

Informativeness: 2: I learned something interesting or useful.

Soundness: 2: I have minor or fixable concerns about the paper's soundness.

### Critical Strengths

1. A technically interesting formulation of implicit flow policies for maximum-entropy reinforcement learning. A major strength of this paper is its integration of implicit numerical methods with expressive flow-based policies. The authors formulate implicit Euler integration as a joint equilibrium system and derive the corresponding policy gradients through implicit differentiation. Furthermore, the discrete change-of-variables formulation enables direct evaluation of the final-action log-density, allowing the resulting policy to be trained within the maximum-entropy SAC framework. This is a meaningful technical contribution because obtaining tractable action densities and stable policy gradients remains challenging for multi-step generative policies. The mathematical derivations are generally well presented, and the paper discusses the assumptions required for their validity.
2. Strong evidence of the benefits of implicit integration for precision-constrained multimodal action distributions. The MultiGoal docking experiments provide encouraging evidence that implicit integration can improve the ability of flow policies to represent multiple precise action modes. DEFlow achieves the highest mean successful coverage across all four evaluated variants, with particularly notable improvements under tight docking constraints. The matched explicit-Euler comparison and step-count analyses further support the argument that numerical integration choices can substantially influence the quality of generated action distributions.

### Critical Weaknesses

1. Insufficient ablation to isolate the contribution of the joint deep-equilibrium formulation from implicit Euler integration. Although the paper introduces both implicit Euler integration and a joint deep-equilibrium formulation, the experimental comparisons do not clearly establish the necessity of solving all intermediate actions jointly. The principal controlled comparison replaces implicit Euler with explicit Euler, demonstrating the benefits of implicit integration. However, this experiment does not distinguish the benefits of implicit Euler discretization from those of the proposed joint equilibrium solver. Since the implicit Euler equations have a sequential structure, an alternative implementation could solve each implicit step independently, potentially using a sequential root solver or Newton iterations. Furthermore, the proposed method introduces additional computational costs, including nonlinear forward solving and dense linear backward solving. Without a comparison against sequential implicit Euler under matched computational budgets, it remains unclear whether the joint equilibrium formulation offers advantages beyond those provided by implicit integration itself.
2. Limited practical performance improvements relative to the computational and numerical costs. Although DEFlow performs well on the precision-constrained MultiGoal docking benchmarks, its advantages are considerably less consistent on standard continuous-control and robotic manipulation tasks. On MuJoCo, DEFlow does not consistently outperform existing methods, and standard SAC achieves a higher aggregate return across the five evaluated tasks. On manipulation benchmarks, DEFlow achieves strong time-averaged returns but does not consistently outperform alternative methods in final task success. For example, SAC achieves substantially higher final success on PegInsertSide, while DIME achieves higher success on PickCube. Several manipulation tasks also exhibit low absolute success rates for DEFlow despite improvements in return. These limitations are particularly relevant given the additional computational requirements of the proposed method. The authors report approximately 40% lower training throughput than SAC-Flow and approximately three times the training time of standard SAC.

### Questions For Authors

How reliable are the policy gradients and action densities when the equilibrium solver does not converge?

The theoretical derivations assume an exact equilibrium solution, while the practical implementation uses finite tolerances, regularization, and clipping. The numerical audit reports that a substantial fraction of evaluation calls reach the solver iteration limit, and some tasks exhibit considerable discrepancies in actor gradients. Could the authors clarify how frequently solver non-convergence occurs during training and quantify its effect on the SAC objective? In particular, how sensitive are policy learning and performance to the solver tolerance, maximum iterations, and Jacobian regularization?

Flag For Ethics Review: No.

Rating: 2: Weak rejection.

Confidence: 1: I do not have sufficient expertise to assess this submission reliably and have alerted the AC.

## Reviewer agWk

Official review dated 19 October 2026, 15:59.

### Paper Summary

The paper proposes DEFlow, a flow policy trained with SAC, based on SAC-Flow. It replaces explicit Euler sampling with implicit Euler and solves all intermediate actions jointly as an equilibrium system. It uses implicit differentiation for policy gradients and derives the final-action density for SAC's entropy term. They try to address a real issue: more denoising steps do not always improve flow-policy performance. Experiments on docking, manipulation, and MuJoCo show benefits mainly in docking and early learning on several manipulation tasks, at additional computational cost.

Presentation: 2: Generally understandable, but requires substantial effort in places.

Informativeness: 2: I learned something interesting or useful.

Soundness: 2: I have minor or fixable concerns about the paper's soundness.

### Critical Strengths

1. The paper points out a real numerical issue in flow policies. Using implicit Euler to improve stability with few sampling steps is a reasonable idea and has some value.
2. The matched explicit-Euler comparison and step-count ablations provide useful evidence that implicit sampling can help on precision tasks. I also appreciate the numerical checks of action densities and gradients.

### Critical Weaknesses

1. The method adds substantial computational cost, with training taking about three times as long as SAC. The gains are mainly in docking and early learning on some manipulation tasks, with no consistent advantage in MuJoCo or final manipulation success. I am not convinced these gains justify the extra computation in practice.
2. I found the theoretical derivations difficult to follow. Although the paper discusses some limitations, it could explain more clearly when implicit sampling is useful and when a simpler explicit sampler would be sufficient.

### Questions For Authors

Could the authors further discuss when DEFlow is worth using? In particular, which types of tasks benefit enough from implicit sampling to justify its additional computational cost compared with explicit sampling?

Flag For Ethics Review: No.

Rating: 3: Weak acceptance.

Confidence: 1: I do not have sufficient expertise to assess this submission reliably and have alerted the AC.
