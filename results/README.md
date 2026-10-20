# Experiment Outputs

- `aggregate_metrics.csv`: 30-seed averages for the four classical baselines.
- `run_metrics.csv`: per-seed baseline metrics.
- `per_process_metrics.csv`: per-process baseline metrics.
- `timelines.csv`: raw baseline execution segments; useful for Gantt charts, but comparatively large.
- `experiment_config.json`: baseline workload/seed/policy settings.
- `plots/`: classical baseline metric comparisons.
- `rl_training_log.csv`: episode reward, epsilon, steps and Q-table state count.
- `rl_q_table.json`: trained tabular Q-table.
- `rl_experiment_config.json`: RL state/action/reward and seed configuration.
- `rl_evaluation_runs.csv`: per-seed metrics for the four baselines and Q-learning.
- `rl_evaluation_per_process.csv`: per-process metrics for all evaluation policies.
- `rl_evaluation_timelines.csv`: execution segments for all evaluation policies.
- `rl_evaluation_aggregate.csv`: aggregate metrics, paired waiting-time deltas, and paired percentile-bootstrap 95% intervals based on 10,000 deterministic resamples of seed-level differences.
- `rl_plots/`: RL training and comparison figures.
- `validate_results.py` (run from the project root) audits run counts, aggregate means, paired policy coverage, process time accounting, and bootstrap interval bounds.

Recreate the baseline outputs with `python benchmark.py --seeds 30 --output results` and `python plot_results.py`. Recreate the RL outputs with `python run_rl_experiment.py --train-episodes 500 --eval-seeds 30 --output results` followed by `python plot_rl_results.py`.

Note: `results_smoke/` folders are quick smoke-test outputs and are not final experimental evidence.
