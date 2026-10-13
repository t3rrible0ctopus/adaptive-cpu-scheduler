"""Train Q-learning and compare it with classical policies on matched workloads.

Defaults: 500 training episodes, 30 evaluation seeds per workload/load cell.
Outputs are written under results/.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from statistics import mean, stdev

from q_learning import QLearningAgent
from rl_env import ACTION_NAMES, RLSchedulingEnv
from scheduler_sim import simulate
from workloads import LOAD_LEVELS, generate_workload

BASELINES = ("FCFS", "SJF", "RR", "PRIORITY")
WORKLOAD_TYPES = ("cpu", "io", "mixed")
LOAD_NAMES = ("light", "medium", "heavy")
METRICS = (
    "average_waiting_time", "average_turnaround_time", "average_response_time",
    "throughput_processes_per_tick", "cpu_utilization", "context_switches",
    "max_waiting_time", "jain_fairness_index_service_rate_proxy", "makespan",
    "scheduling_decisions", "decision_overhead_total",
)


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def train_agent(episodes: int, seed: int, output_dir: str,
                switch_penalty: float, starvation_penalty: float) -> QLearningAgent:
    rng = random.Random(seed)
    agent = QLearningAgent(alpha=0.12, gamma=0.95, seed=seed)
    log = []
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.992
    for episode in range(episodes):
        kind = rng.choice(WORKLOAD_TYPES)
        load = rng.choice(LOAD_NAMES)
        workload_seed = seed * 100_000 + episode
        workload = generate_workload(kind, load, seed=workload_seed)
        env = RLSchedulingEnv(workload, switch_penalty=switch_penalty,
                              starvation_penalty=starvation_penalty)
        state = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            action = agent.choose_action(state, epsilon=epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            if steps > 100_000:
                raise RuntimeError("Training episode exceeded safety step limit")
        log.append({
            "episode": episode + 1, "workload_type": kind, "load_level": load,
            "seed": workload_seed, "steps": steps, "total_reward": total_reward,
            "epsilon": epsilon, "q_states_seen": len(agent.q),
        })
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    write_csv(os.path.join(output_dir, "rl_training_log.csv"), log)
    agent.save(os.path.join(output_dir, "rl_q_table.json"))
    return agent


def run_agent(agent: QLearningAgent, workload, overhead: float = 0.0,
              switch_penalty: float = 0.15, starvation_penalty: float = 0.75):
    env = RLSchedulingEnv(workload, decision_overhead_per_decision=overhead,
                          switch_penalty=switch_penalty,
                          starvation_penalty=starvation_penalty)
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state, epsilon=0.0)
        state, _, done, _ = env.step(action)
    return env.result()


def evaluate(agent: QLearningAgent, eval_seeds: int, seed_start: int,
             output_dir: str, quantum: int, overhead: float,
             switch_penalty: float, starvation_penalty: float) -> list[dict]:
    rows = []
    per_process_rows = []
    timeline_rows = []
    for kind in WORKLOAD_TYPES:
        for load in LOAD_NAMES:
            for i in range(eval_seeds):
                seed = seed_start + i
                workload = generate_workload(kind, load, seed=seed)
                for policy in BASELINES:
                    result = simulate(workload, policy, quantum=quantum,
                                      decision_overhead_per_decision=overhead)
                    rows.append({"workload_type": kind, "load_level": load,
                                 "seed": seed, **result.summary()})
                    for p in result.per_process:
                        per_process_rows.append({"workload_type": kind, "load_level": load,
                                                 "seed": seed, "policy": policy, **p})
                    for seg in result.timeline:
                        timeline_rows.append({"workload_type": kind, "load_level": load,
                                              "seed": seed, "policy": policy, **seg})
                result = run_agent(agent, workload, overhead=overhead,
                                   switch_penalty=switch_penalty,
                                   starvation_penalty=starvation_penalty)
                rows.append({"workload_type": kind, "load_level": load,
                             "seed": seed, **result.summary()})
                for p in result.per_process:
                    per_process_rows.append({"workload_type": kind, "load_level": load,
                                             "seed": seed, "policy": "QLEARNING", **p})
                for seg in result.timeline:
                    timeline_rows.append({"workload_type": kind, "load_level": load,
                                          "seed": seed, "policy": "QLEARNING", **seg})
    write_csv(os.path.join(output_dir, "rl_evaluation_runs.csv"), rows)
    write_csv(os.path.join(output_dir, "rl_evaluation_per_process.csv"), per_process_rows)
    write_csv(os.path.join(output_dir, "rl_evaluation_timelines.csv"), timeline_rows)
    return rows


def aggregate(rows: list[dict], output_dir: str) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["workload_type"], row["load_level"], row["policy"])].append(row)
    aggregates = []
    for (kind, load, policy), group in sorted(groups.items()):
        row = {"workload_type": kind, "load_level": load, "policy": policy,
               "runs": len(group)}
        for metric in METRICS:
            row[metric] = mean(float(x[metric]) for x in group)
        aggregates.append(row)

    # Paired confidence intervals for waiting-time differences against RR.
    by_cell = defaultdict(dict)
    for row in rows:
        by_cell[(row["workload_type"], row["load_level"], row["seed"])][row["policy"]] = row
    cells = defaultdict(list)
    for (kind, load, seed), policies in by_cell.items():
        if "RR" in policies and "QLEARNING" in policies:
            rr = float(policies["RR"]["average_waiting_time"])
            rl = float(policies["QLEARNING"]["average_waiting_time"])
            cells[(kind, load)].append((rr, rl, seed))
    for row in aggregates:
        if row["policy"] != "QLEARNING":
            continue
        pairs = cells[(row["workload_type"], row["load_level"])]
        deltas = [rl - rr for rr, rl, _ in pairs]
        rr_means = [rr for rr, _, _ in pairs]
        mean_delta = mean(deltas)
        sd = stdev(deltas) if len(deltas) > 1 else 0.0
        half_width = 1.96 * sd / math.sqrt(len(deltas)) if deltas else 0.0
        row["mean_waiting_delta_vs_RR"] = mean_delta
        row["relative_waiting_improvement_vs_RR_pct"] = (
            100.0 * (mean(rr_means) - mean([rl for _, rl, _ in pairs])) / mean(rr_means)
            if rr_means and mean(rr_means) else 0.0
        )
        row["paired_waiting_delta_95ci_low"] = mean_delta - half_width
        row["paired_waiting_delta_95ci_high"] = mean_delta + half_width
    write_csv(os.path.join(output_dir, "rl_evaluation_aggregate.csv"), aggregates)
    return aggregates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--eval-seeds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed-start", type=int, default=1000)
    parser.add_argument("--quantum", type=int, default=3)
    parser.add_argument("--overhead", type=float, default=0.0,
                        help="Accounting only; not injected into simulated time")
    parser.add_argument("--switch-penalty", type=float, default=0.15)
    parser.add_argument("--starvation-penalty", type=float, default=0.75)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    if args.train_episodes <= 0 or args.eval_seeds <= 0:
        parser.error("--train-episodes and --eval-seeds must be positive")
    os.makedirs(args.output, exist_ok=True)
    config = {
        "algorithm": "tabular Q-learning",
        "training_episodes": args.train_episodes,
        "evaluation_seeds_per_workload_load_cell": args.eval_seeds,
        "training_seed": args.seed,
        "evaluation_seed_start": args.eval_seed_start,
        "baseline_quantum": args.quantum,
        "decision_overhead_per_decision_accounting_only": args.overhead,
        "actions": list(ACTION_NAMES),
        "state_features": ["ready queue size bin", "average waiting bin", "maximum waiting bin",
                            "minimum next CPU burst bin", "average next CPU burst bin",
                            "I/O-bound ready fraction bin", "high-priority ready fraction bin",
                            "number of I/O-blocked processes bin", "last process still ready flag"],
        "reward": "-number of other ready processes waiting this tick - switch_penalty on PID switch - starvation_penalty per ready process at/above waiting threshold",
        "switch_penalty": args.switch_penalty,
        "starvation_penalty": args.starvation_penalty,
        "starvation_threshold_ticks": 10,
        "workload_types": list(WORKLOAD_TYPES),
        "load_levels": list(LOAD_NAMES),
        "baselines": list(BASELINES),
        "note": "One tick per RL action; learned policy is preemptive at tick boundaries. Synthetic workloads only.",
    }
    with open(os.path.join(args.output, "rl_experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    agent = train_agent(args.train_episodes, args.seed, args.output,
                        args.switch_penalty, args.starvation_penalty)
    rows = evaluate(agent, args.eval_seeds, args.eval_seed_start,
                    args.output, args.quantum, args.overhead,
                    args.switch_penalty, args.starvation_penalty)
    aggregates = aggregate(rows, args.output)
    q_rows = [r for r in aggregates if r["policy"] == "QLEARNING"]
    print(f"Trained for {args.train_episodes} episodes; visited {len(agent.q)} states.")
    print(f"Evaluated {len(rows)} matched policy runs across {len(q_rows)} RL workload/load cells.")
    print(f"Aggregate results: {os.path.join(args.output, 'rl_evaluation_aggregate.csv')}")
    print("RL average waiting-time improvement vs RR by workload/load:")
    for row in q_rows:
        print(f"  {row['workload_type']:5s} {row['load_level']:6s}: "
              f"{row['relative_waiting_improvement_vs_RR_pct']:+.2f}% "
              f"(paired 95% CI for RL-RR: {row['paired_waiting_delta_95ci_low']:.2f}, "
              f"{row['paired_waiting_delta_95ci_high']:.2f})")


if __name__ == "__main__":
    main()
