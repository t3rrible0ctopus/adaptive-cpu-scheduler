# Project Implementation Report

## 1. Project objective

Build and evaluate a single-core CPU scheduling simulator that compares classical scheduling policies with a reinforcement-learning-based adaptive scheduler under synthetic CPU-bound, I/O-bound and mixed workloads. The paper's proposed 15% reduction in average waiting time against Round Robin is treated as a hypothesis to test, not a guaranteed outcome.

## 2. Implementation components

- `scheduler_sim.py`: baseline discrete-time simulator; FCFS, non-preemptive SJF, Round Robin and preemptive priority; process metrics and execution timeline.
- `workloads.py`: seeded generation of CPU-bound, I/O-bound and mixed workloads at light, medium and heavy arrival rates.
- `benchmark.py`: paired classical-baseline experiments and CSV export.
- `rl_env.py`: reset/step interface for a single-core RL environment with arrivals, I/O blocking, a ready queue, timeline and metrics.
- `q_learning.py`: tabular Q-learning with epsilon-greedy exploration, Bellman updates, and JSON save/load.
- `run_rl_experiment.py`: trains one agent across randomly sampled workload types/load levels and evaluates it against all four baselines on matched held-out seeds.
- `plot_results.py`, `plot_rl_results.py`: baseline and RL comparison plots.
- `tests/`: unit tests for baseline scheduling and the RL environment/agent.

## 3. RL formulation

### State

A hashable tuple of discretized features: ready-queue size, average and maximum ready wait, minimum and average current CPU-burst lengths, fraction of I/O-bound ready processes, fraction of high-priority ready processes, count of I/O-blocked processes, and whether the last process is still ready.

### Actions

The agent selects one of six dispatch heuristics: oldest ready, shortest current CPU burst, highest priority, longest waiting, shortest total remaining CPU demand, or continue the last process if still ready. The selected heuristic chooses the process that executes for one tick. This fixed action space is more tractable for tabular Q-learning than a variable-size action for every process ID.

### Reward

Per tick, the environment penalizes the number of other ready processes that wait, a process-ID switch, and each ready process that exceeds the starvation-wait threshold. The weights are configurable and written to `results/rl_experiment_config.json`.

### Learning

Tabular Q-learning uses epsilon-greedy exploration, learning rate 0.12 and discount factor 0.95. Training episodes sample workload type and load level. Evaluation uses a separate seed range and applies the learned policy with exploration disabled.

## 4. Experimental design

### Classical baselines

The stored baseline run uses 30 seeds for each of 3 workload types and 3 load levels, comparing 4 policies: 3 × 3 × 30 × 4 = 1,080 runs. Policies receive identical generated workloads within each seed/cell.

### RL evaluation

`run_rl_experiment.py` defaults to 500 training episodes and 30 evaluation seeds for each of the 9 workload/load cells. Each evaluation workload is run with FCFS, SJF, RR, priority and Q-learning, so the policy comparisons are paired. It exports per-run and aggregate metrics plus paired waiting-time deltas and approximate 95% confidence intervals.

## 5. Metrics

Average waiting, turnaround and response time; throughput; CPU utilization; context switches; maximum waiting time; Jain fairness index over CPU-demand/turnaround service-rate proxy; makespan; number of scheduling decisions; and accounting-only decision overhead.

## 6. Important limitations

1. Workloads are synthetic, not real OS traces.
2. SJF is non-preemptive and is given the next CPU burst in advance.
3. The learned policy makes one decision per tick and can preempt at tick boundaries. It is a simulator policy, not a Linux kernel implementation.
4. The action is a dispatch heuristic that selects the next process indirectly, keeping the tabular action space finite.
5. Decision overhead is currently counted separately and is not added to simulated execution time; it is not measured wall-clock inference latency.
6. Fairness is reported using a service-rate proxy and maximum waiting time; the proxy needs further justification before a strong fairness claim.
7. The confidence intervals use a normal 1.96 multiplier on paired differences and are approximate, especially for small evaluation samples.
8. Tabular Q-learning can struggle with state-space sparsity; a DQN or feature-based policy may be appropriate in a later iteration.

## 7. Reproducibility

Python 3.10+ is recommended. Install dependencies with `pip install -r requirements.txt`. Run unit tests before experiments. Training seed, evaluation seed range, actions, state features, reward weights and policy parameters are written to the result configuration files. Exact results may change if code or parameters change.
## 8. Preliminary measured results

The committed results use 500 Q-learning training episodes and 30 held-out evaluation seeds per workload/load cell. There are 1,350 matched evaluation runs (9 workload/load cells × 30 seeds × 5 policies). Full tables are in `RESULTS_SUMMARY.md` and `results/rl_evaluation_aggregate.csv`.

The learned policy reduced mean waiting time relative to Round Robin in all nine cells, by 5.91%–26.90% (unweighted mean of cell-level percentage reductions: 17.78%). The paper's 15% target was reached in five of nine cells; all three I/O-bound cells and the mixed/light cell were below 15%. Q-learning did not beat the idealized SJF baseline on mean waiting time in any cell. It also had more context switches than Round Robin and a generally lower Jain service-rate fairness proxy. Therefore, the initial results are promising but do not demonstrate that the complete research objective has been met.

The approximate paired 95% intervals for the Q-learning minus Round Robin waiting-time differences are below zero in all nine cells. They use a normal 1.96 multiplier; a stronger final paper should confirm results with a paired t-interval or bootstrap, report effect sizes, and predefine the analysis protocol.

Training reward improved on average between the first and last 50 episodes, from about -4,164.28 to -3,611.45. This is a training diagnostic only; it does not establish convergence.

## 9. Next research tasks

1. Tune reward weights using a validation seed set, then reserve a new test seed set for final reporting.
2. Investigate why I/O-bound workloads miss the 15% waiting-time target.
3. Reduce context switches and measure the trade-off against waiting time.
4. Improve fairness evaluation, including starvation counts and a better-justified Jain index input.
5. Measure actual Python inference latency and/or model scheduling overhead in simulation time.
6. Add a paired bootstrap or t-based confidence interval and ablation studies for reward components.
7. Consider DQN only if the tabular state representation proves insufficient.

