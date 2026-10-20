"""Create review-ready plots for the RL-vs-baseline evaluation."""
from __future__ import annotations
import argparse
import os
import time
import pandas as pd
import matplotlib.pyplot as plt


def save_figure(figure, path: str) -> None:
    """Retry transient Windows file-open failures while replacing plot files."""
    for attempt in range(4):
        try:
            figure.savefig(path, dpi=160)
            return
        except OSError as exc:
            if attempt == 3:
                raise OSError(f"Could not save plot {path!r} after 4 attempts: {exc}") from exc
            time.sleep(0.25 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", default="results/rl_evaluation_aggregate.csv")
    parser.add_argument("--training", default="results/rl_training_log.csv")
    parser.add_argument("--output", default="results/rl_plots")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    output_dir = os.path.abspath(args.output)
    df = pd.read_csv(args.aggregate)
    metrics = [
        ("average_waiting_time", "Average waiting time (ticks)"),
        ("average_turnaround_time", "Average turnaround time (ticks)"),
        ("average_response_time", "Average response time (ticks)"),
        ("cpu_utilization", "CPU utilization"),
        ("max_waiting_time", "Maximum waiting time (ticks)"),
        ("jain_fairness_index_service_rate_proxy", "Jain fairness index (service-rate proxy)"),
        ("context_switches", "Context switches"),
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
            save_figure(plt.gcf(), os.path.join(output_dir, f"{metric}_{workload}.png"))
            plt.close()
    if os.path.exists(args.training):
        training = pd.read_csv(args.training)
        training["rolling_reward"] = training["total_reward"].rolling(25, min_periods=1).mean()
        ax = training.plot(x="episode", y="rolling_reward", legend=False,
                           title="Q-learning training reward (25-episode rolling mean)")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Rolling mean episode reward")
        plt.tight_layout()
        save_figure(plt.gcf(), os.path.join(output_dir, "training_reward.png"))
        plt.close()
    print(f"Saved RL plots to {output_dir}")


if __name__ == "__main__":
    main()
