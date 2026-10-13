"""Tabular Q-learning agent for RLSchedulingEnv."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional


from rl_env import N_ACTIONS


class QLearningAgent:
    def __init__(self, alpha: float = 0.12, gamma: float = 0.95, seed: int = 42):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.rng = random.Random(seed)
        self.q: dict[tuple, list[float]] = {}

    def values(self, state: tuple) -> list[float]:
        if state not in self.q:
            self.q[state] = [0.0] * N_ACTIONS
        return self.q[state]

    def choose_action(self, state: tuple, epsilon: float = 0.0) -> int:
        if self.rng.random() < epsilon:
            return self.rng.randrange(N_ACTIONS)
        # Do not create Q-table entries during evaluation for unseen states.
        # New states are materialized by update() during training only.
        values = self.q.get(state, [0.0] * N_ACTIONS)
        best = max(values)
        candidates = [i for i, value in enumerate(values) if value == best]
        # Greedy evaluation is deterministic; exploration/training can randomize ties.
        return candidates[0] if epsilon <= 0 else self.rng.choice(candidates)

    def update(self, state: tuple, action: int, reward: float,
               next_state: tuple, done: bool) -> None:
        q_values = self.values(state)
        target = reward if done else reward + self.gamma * max(self.values(next_state))
        q_values[action] += self.alpha * (target - q_values[action])

    def save(self, path: str | Path) -> None:
        payload = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "n_actions": N_ACTIONS,
            "q_table": {json.dumps(list(state)): values for state, values in self.q.items()},
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, seed: int = 42) -> "QLearningAgent":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("n_actions") != N_ACTIONS:
            raise ValueError("Saved Q-table action count does not match this environment")
        agent = cls(alpha=payload["alpha"], gamma=payload["gamma"], seed=seed)
        agent.q = {tuple(json.loads(state)): list(values)
                   for state, values in payload["q_table"].items()}
        return agent
