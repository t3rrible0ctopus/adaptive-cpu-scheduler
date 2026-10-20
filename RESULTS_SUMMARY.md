# Preliminary RL Evaluation Summary

**Status: reproduced simulator evidence, not a final real-system research conclusion.** One tabular Q-learning agent was trained for 500 episodes across randomly sampled synthetic workload types and load levels. It was evaluated on 30 held-out seeds in each of 9 workload/load cells. Within each cell and seed, the same generated workload was run with FCFS, SJF, Round Robin, Priority and Q-learning: 9 × 30 × 5 = **1,350 evaluation runs**. The committed run was regenerated from the documented command and matched all stored waiting-time means; baseline and RL aggregate values were also checked against their per-run CSVs.

## Average waiting time and comparison with Round Robin

| Workload | Load | RR waiting | Q-learning waiting | Improvement vs RR | Paired bootstrap 95% CI for (QL − RR) waiting | SJF waiting |
|---|---|---:|---:|---:|---:|---:|
| CPU | light | 86.11 | 66.72 | 22.52% | [-22.09, -16.83] | 41.16 |
| CPU | medium | 201.42 | 154.43 | 23.33% | [-54.89, -38.97] | 97.75 |
| CPU | heavy | 333.04 | 255.41 | 23.31% | [-87.55, -66.73] | 165.80 |
| IO | light | 32.47 | 29.16 | 10.18% | [-4.16, -2.45] | 26.50 |
| IO | medium | 108.09 | 94.23 | 12.82% | [-15.64, -12.10] | 82.80 |
| IO | heavy | 197.94 | 186.25 | 5.91% | [-13.56, -9.76] | 154.22 |
| MIXED | light | 60.52 | 53.28 | 11.96% | [-9.29, -5.41] | 36.25 |
| MIXED | medium | 159.48 | 122.64 | 23.10% | [-41.44, -32.15] | 83.42 |
| MIXED | heavy | 265.95 | 194.41 | 26.90% | [-78.77, -63.91] | 144.98 |

## Interpretation

- Q-learning had lower mean waiting time than Round Robin in all nine evaluated cells. The observed reduction ranges from **5.91% to 26.90%**; the unweighted mean of the nine cell-level percentage reductions is **17.78%**.
- The paper’s 15% target is reached in **5 of 9** cells, not all cells. All three I/O-bound cells and mixed/light miss the target. Do not claim the 15% objective has been met universally.
- Q-learning does **not** beat the idealized SJF baseline on average waiting time in any of the nine cells. SJF is non-preemptive but knows the next CPU burst in advance, making it an optimistic reference point.
- Q-learning causes more context switches than Round Robin in every cell, and its Jain service-rate fairness proxy is lower in every cell. The current prototype therefore shows a waiting-time trade-off rather than a complete solution to the paper’s combined waiting/fairness/overhead objective.
- The paired percentile-bootstrap 95% intervals for Q-learning minus Round Robin waiting-time differences are below zero in all nine cells. They use 10,000 resamples of the 30 paired per-seed differences, with deterministic cell-specific seeds. They quantify uncertainty conditional on this one trained agent and these synthetic workloads; they do not capture variation across training seeds or workload models.
- Decision overhead is accounting-only and was set to zero in this experiment. It is not measured wall-clock inference latency and is not included in simulated time.

## Training diagnostics

- Training episodes: 500.
- Distinct tabular states updated during training: 4,768.
- Mean episode reward over the first 50 episodes: approximately -4,164.28.
- Mean episode reward over the last 50 episodes: approximately -3,611.45 (less negative is better for this reward).
- Reward improvement is an encouraging training diagnostic, not proof of policy convergence or generalization.

## Conclusion for the review

The project has a working RL environment, a trainable tabular Q-learning agent, held-out matched evaluation, reproducible CSV outputs, and comparison plots. This experiment supports a limited claim: the prototype reduced average waiting time versus Round Robin on these synthetic workloads. It does not satisfy the 15% target in every workload, outperform SJF, or establish fairness/overhead benefits. These limitations should be shown clearly to the reviewer.
