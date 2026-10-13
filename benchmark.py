"""Run paired benchmark experiments and export CSV summaries."""
from __future__ import annotations
import argparse
import csv
import json
import os
from statistics import mean
from scheduler_sim import simulate
from workloads import generate_workload, LOAD_LEVELS

POLICIES = ["FCFS", "SJF", "RR", "PRIORITY"]


def run_benchmarks(output_dir="results", seeds=range(30), quantum=3, overhead=0.0):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    process_rows = []
    timeline_rows = []
    for kind in ("cpu", "io", "mixed"):
        for load in ("light", "medium", "heavy"):
            for seed in seeds:
                workload = generate_workload(kind, load, seed=seed)
                # Each policy receives exactly the same workload for this seed.
                for policy in POLICIES:
                    result = simulate(
                        workload, policy, quantum=quantum,
                        decision_overhead_per_decision=overhead
                    )
                    rows.append({
                        "workload_type": kind, "load_level": load, "seed": seed,
                        **result.summary()
                    })
                    for p in result.per_process:
                        process_rows.append({
                            "workload_type": kind, "load_level": load,
                            "seed": seed, "policy": policy, **p
                        })
                    for seg in result.timeline:
                        timeline_rows.append({
                            "workload_type": kind, "load_level": load,
                            "seed": seed, "policy": policy, **seg
                        })

    def write_csv(path, data):
        if not data:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)

    write_csv(os.path.join(output_dir, "run_metrics.csv"), rows)
    write_csv(os.path.join(output_dir, "per_process_metrics.csv"), process_rows)
    write_csv(os.path.join(output_dir, "timelines.csv"), timeline_rows)

    # Mean metrics across seeds, grouped by workload and policy.
    keys = [
        "average_waiting_time", "average_turnaround_time", "average_response_time",
        "throughput_processes_per_tick", "cpu_utilization", "context_switches",
        "max_waiting_time", "jain_fairness_index_service_rate_proxy",
        "scheduling_decisions", "decision_overhead_total", "makespan"
    ]
    grouped = {}
    for row in rows:
        key = (row["workload_type"], row["load_level"], row["policy"])
        grouped.setdefault(key, []).append(row)
    aggregate = []
    for (kind, load, policy), group in sorted(grouped.items()):
        out = {"workload_type": kind, "load_level": load, "policy": policy,
               "runs": len(group)}
        for metric in keys:
            out[metric] = mean(float(r[metric]) for r in group)
        aggregate.append(out)
    write_csv(os.path.join(output_dir, "aggregate_metrics.csv"), aggregate)

    with open(os.path.join(output_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "seeds": list(seeds), "quantum": quantum,
            "decision_overhead_per_decision": overhead,
            "policies": POLICIES, "workload_types": ["cpu", "io", "mixed"],
            "load_levels": list(LOAD_LEVELS.keys()),
        }, f, indent=2)
    return rows, aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results")
    parser.add_argument("--seeds", type=int, default=30,
                        help="Number of matched seeds; use 1 for a quick smoke test")
    parser.add_argument("--quantum", type=int, default=3)
    parser.add_argument("--overhead", type=float, default=0.0,
                        help="Accounting cost per scheduling decision (reported separately)")
    args = parser.parse_args()
    rows, aggregate = run_benchmarks(
        args.output, seeds=range(args.seeds), quantum=args.quantum, overhead=args.overhead
    )
    print(f"Completed {len(rows)} policy runs.")
    print(f"Aggregate results: {os.path.join(args.output, 'aggregate_metrics.csv')}")


if __name__ == "__main__":
    main()
