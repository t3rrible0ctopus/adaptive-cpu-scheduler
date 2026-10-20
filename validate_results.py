"""Check committed experiment outputs for completeness and internal consistency."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


BASELINE_METRICS = (
    "average_waiting_time", "average_turnaround_time", "average_response_time",
    "throughput_processes_per_tick", "cpu_utilization", "context_switches",
    "max_waiting_time", "jain_fairness_index_service_rate_proxy",
    "scheduling_decisions", "decision_overhead_total", "makespan",
)
RL_METRICS = BASELINE_METRICS
POLICIES = {"FCFS", "SJF", "RR", "PRIORITY", "QLEARNING"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify_aggregates(run_rows: list[dict[str, str]], aggregate_rows: list[dict[str, str]],
                      metrics: tuple[str, ...], label: str) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in run_rows:
        groups[(row["workload_type"], row["load_level"], row["policy"])].append(row)
    if len(groups) != len(aggregate_rows):
        raise ValueError(f"{label}: aggregate group count does not match run data")
    for aggregate in aggregate_rows:
        key = (aggregate["workload_type"], aggregate["load_level"], aggregate["policy"])
        rows = groups.get(key)
        if not rows:
            raise ValueError(f"{label}: missing run group {key}")
        if int(aggregate["runs"]) != len(rows):
            raise ValueError(f"{label}: incorrect run count for {key}")
        for metric in metrics:
            expected = mean(float(row[metric]) for row in rows)
            actual = float(aggregate[metric])
            if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"{label}: {key} {metric} does not match run data")


def verify_process_accounting(rows: list[dict[str, str]], label: str) -> None:
    for row in rows:
        turnaround = float(row["turnaround_time"])
        accounted = (float(row["waiting_time"]) + float(row["cpu_time"])
                     + float(row["io_time"]))
        if not math.isclose(turnaround, accounted, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"{label}: process {row['pid']} has inconsistent time accounting")


def verify_paired_runs(rows: list[dict[str, str]]) -> None:
    groups: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        groups[(row["workload_type"], row["load_level"], row["seed"])].add(row["policy"])
    expected = POLICIES
    if not groups or any(policies != expected for policies in groups.values()):
        raise ValueError("RL run data does not contain all five policies for every paired seed")
    if any(len(seeds) != 30 for seeds in _seeds_by_cell(groups).values()):
        raise ValueError("RL run data does not contain 30 paired seeds in every cell")


def _seeds_by_cell(groups: dict[tuple[str, str, str], set[str]]) -> dict[tuple[str, str], set[str]]:
    result: dict[tuple[str, str], set[str]] = defaultdict(set)
    for workload, load, seed in groups:
        result[(workload, load)].add(seed)
    return result


def validate_results(results_dir: Path) -> tuple[int, int, int]:
    baseline_runs = read_csv(results_dir / "run_metrics.csv")
    baseline_aggregate = read_csv(results_dir / "aggregate_metrics.csv")
    rl_runs = read_csv(results_dir / "rl_evaluation_runs.csv")
    rl_aggregate = read_csv(results_dir / "rl_evaluation_aggregate.csv")
    baseline_processes = read_csv(results_dir / "per_process_metrics.csv")
    rl_processes = read_csv(results_dir / "rl_evaluation_per_process.csv")
    training = read_csv(results_dir / "rl_training_log.csv")

    if len(baseline_runs) != 1080 or len(rl_runs) != 1350 or len(training) != 500:
        raise ValueError("Unexpected baseline, RL evaluation, or training row count")
    verify_aggregates(baseline_runs, baseline_aggregate, BASELINE_METRICS, "baseline")
    verify_aggregates(rl_runs, rl_aggregate, RL_METRICS, "RL")
    verify_process_accounting(baseline_processes, "baseline")
    verify_process_accounting(rl_processes, "RL")
    verify_paired_runs(rl_runs)
    for row in rl_aggregate:
        if row["policy"] == "QLEARNING":
            low = float(row["paired_waiting_delta_bootstrap_95ci_low"])
            high = float(row["paired_waiting_delta_bootstrap_95ci_high"])
            if not math.isfinite(low) or not math.isfinite(high) or low > high:
                raise ValueError("Invalid paired bootstrap interval")
    return len(baseline_runs), len(rl_runs), len(training)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    baseline, rl, training = validate_results(args.results)
    print(f"Validated {baseline} baseline runs, {rl} RL evaluation runs, and {training} episodes.")
    print("Aggregate means, process time accounting, paired seeds, and bootstrap intervals are consistent.")


if __name__ == "__main__":
    main()
