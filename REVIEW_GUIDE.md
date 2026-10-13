# Second Review Guide

## What to demonstrate (5–8 minutes)

1. **Scope (30 seconds):** Single-core CPU scheduling in a discrete-time Python simulator; this is not a kernel-level scheduler.
2. **Classical baselines (1 minute):** Explain FCFS, non-preemptive SJF, Round Robin (quantum 3), and preemptive priority (smaller number means higher priority).
3. **Validation (1 minute):** Run `python -m unittest discover -s tests -v`; the tests cover known schedules, waiting/turnaround/response metrics, idle time, I/O, invalid inputs and the RL environment contract.
4. **Workload and baseline results (1 minute):** Show `results/aggregate_metrics.csv` and baseline plots in `results/plots/`. The baseline uses 30 matched seeds per workload/load cell.
5. **RL design (1–2 minutes):** Show `rl_env.py`, `q_learning.py`, and `results/rl_experiment_config.json`. The agent uses discretized queue statistics and selects a dispatch heuristic every tick.
6. **RL comparison (1–2 minutes):** Show `results/rl_evaluation_aggregate.csv`, the waiting-time graph and training reward plot. Explain that this is an experimental prototype and the measured outcome—not the 15% target—determines the conclusion.
7. **Limitations / next work (30 seconds):** Synthetic workloads only; tabular Q-learning is limited; decision overhead is accounting-only; fairness is currently a service-rate proxy.

## Commands

```bash
python -m unittest discover -s tests -v
python benchmark.py --seeds 30 --quantum 3 --output results
python plot_results.py --input results/aggregate_metrics.csv --output results/plots
python run_rl_experiment.py --train-episodes 500 --eval-seeds 30 --output results
python plot_rl_results.py
```

## Claims to avoid

- Do not say the agent runs inside Linux or replaces a real OS scheduler.
- Do not claim a 15% improvement unless the evaluation CSV supports it across the stated workload conditions.
- Do not call the accounting-only overhead a measured wall-clock inference cost.
- Do not describe SJF as a realistic online algorithm here: it is given the next CPU burst length in advance.
- Do not call the Jain service-rate proxy a definitive fairness measure without further validation.
