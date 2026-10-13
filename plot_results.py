"""Plot aggregate benchmark results. Requires pandas and matplotlib."""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/aggregate_metrics.csv")
    parser.add_argument("--output", default="results/plots")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    df = pd.read_csv(args.input)

    metrics = [
        ("average_waiting_time", "Average waiting time (ticks)"),
        ("average_turnaround_time", "Average turnaround time (ticks)"),
        ("average_response_time", "Average response time (ticks)"),
        ("cpu_utilization", "CPU utilization"),
        ("context_switches", "Context switches"),
        ("max_waiting_time", "Maximum waiting time (ticks)"),
        ("jain_fairness_index_service_rate_proxy", "Jain fairness index (service-rate proxy)"),
        ("decision_overhead_total", "Accounted scheduling overhead"),
    ]
    for metric, label in metrics:
        for workload in sorted(df["workload_type"].unique()):
            subset = df[df["workload_type"] == workload]
            pivot = subset.pivot(index="load_level", columns="policy", values=metric)
            pivot = pivot.reindex(["light", "medium", "heavy"])
            ax = pivot.plot(kind="bar", title=f"{label} — {workload} workload")
            ax.set_xlabel("Load level")
            ax.set_ylabel(label)
            plt.tight_layout()
            safe = metric.replace("/", "_")
            plt.savefig(os.path.join(args.output, f"{safe}_{workload}.png"), dpi=160)
            plt.close()


if __name__ == "__main__":
    main()
