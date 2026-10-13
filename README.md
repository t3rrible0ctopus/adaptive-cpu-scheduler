# Adaptive Reinforcement Learning-Based CPU Scheduler

A reproducible Python research prototype for comparing classical CPU scheduling algorithms with a tabular Q-learning-based adaptive dispatcher. The simulator models a **single CPU core** with integer simulation ticks; it does not modify or replace a real operating-system kernel.

## Features

- Classical baselines: FCFS, non-preemptive SJF, Round Robin and preemptive Priority.
- Synthetic CPU-bound, I/O-bound and mixed workloads at light, medium and heavy arrival rates.
- Paired baseline benchmarks using identical workloads per seed.
- Metrics: waiting, turnaround and response time, throughput, CPU utilization, context switches, maximum waiting time, a documented Jain service-rate fairness proxy, scheduling decision count and accounting-only decision overhead.
- RL environment with `reset()` / `step()` interface, a discretized observation, six dispatch-heuristic actions, reward shaping for waiting/switching/starvation, and tabular Q-learning.
- Matched RL evaluation against the four classical policies, paired waiting-time deltas, approximate confidence intervals, CSV exports and plots.
- Unit tests for baseline algorithms, edge cases, environment transitions and Q-table persistence.

## Requirements and setup

Python 3.10+ is recommended. The simulator and RL agent use the Python standard library; pandas and matplotlib are used for aggregate analysis and plots.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Reproduce the classical baseline experiment

```bash
python benchmark.py --seeds 30 --quantum 3 --output results
python plot_results.py --input results/aggregate_metrics.csv --output results/plots
```

The included initial baseline dataset covers 3 workload types × 3 load levels × 30 seeds × 4 policies = **1,080 runs**. Policies receive the same workload within each seed/cell. Baseline outputs are stored in `results/`.

For a quick smoke test, use `--seeds 1 --output results_smoke`; smoke-test results are not final research evidence.

## Train and evaluate Q-learning

```bash
python run_rl_experiment.py --train-episodes 500 --eval-seeds 30 --seed 42 --eval-seed-start 1000 --output results
python plot_rl_results.py
```

The script trains one agent across randomly sampled workload types and load levels, then evaluates it on a separate seed range. Each evaluation workload is run with FCFS, SJF, RR, Priority and Q-learning. It writes:

- `results/rl_training_log.csv`
- `results/rl_q_table.json`
- `results/rl_experiment_config.json`
- `results/rl_evaluation_runs.csv`
- `results/rl_evaluation_per_process.csv`
- `results/rl_evaluation_timelines.csv`
- `results/rl_evaluation_aggregate.csv`
- `results/rl_plots/`

A shorter development run is available with `--train-episodes 10 --eval-seeds 1 --output results_smoke_rl`. Do not use smoke-test results to support research claims.

## RL design

### State

The observation is a tuple of discretized ready-queue size, average/max waiting time, minimum/average current CPU burst, I/O-bound fraction in the ready queue, high-priority fraction, number of I/O-blocked processes, and whether the last process is still ready.

### Actions

At each simulation tick, Q-learning selects one of six dispatch heuristics: oldest ready, shortest current CPU burst, highest priority, longest waiting, shortest total remaining CPU demand, or continue the previous process if it remains ready. The selected heuristic chooses which process runs for one tick. This fixed action space makes tabular Q-learning practical. Because decisions occur each tick, the learned policy is preemptive at tick boundaries; the heuristic actions are not identical to the classical non-preemptive algorithms.

### Reward

The environment penalizes the number of other ready processes that wait during the tick, adds a configurable penalty when the selected process differs from the previous process, and penalizes ready processes whose waiting time exceeds a starvation threshold. The settings are recorded in `results/rl_experiment_config.json`.

## Important limitations

1. **Synthetic workloads:** results do not establish generalization to real operating-system traces.
2. **Idealized SJF:** the simulator knows the next CPU burst in advance.
3. **RL action semantics:** the learned policy chooses a dispatch heuristic each tick, not a Linux kernel scheduling class.
4. **Overhead:** decision overhead is currently accounted for separately; it is not inserted into simulated time and is not wall-clock inference latency.
5. **Fairness:** Jain's index uses CPU-demand/turnaround as a service-rate proxy. It must be labelled as such and validated before making strong fairness claims.
6. **Statistical intervals:** paired 95% intervals use a normal approximation; report this limitation and consider a paired bootstrap or t-interval in final analysis.
7. **Tabular state space:** state aggregation is coarse and may still limit generalization. DQN or other function approximation is a possible extension, not part of the current implementation.
8. **Research target:** the proposed 15% reduction in average waiting time versus Round Robin is a hypothesis, not a guaranteed outcome.

See `PROJECT_REPORT.md` for implementation details and `REVIEW_GUIDE.md` for a short demonstration plan.
