import unittest
from scheduler_sim import ProcessSpec
from rl_env import ACTION_NAMES, N_ACTIONS, RLSchedulingEnv
from q_learning import QLearningAgent
from run_rl_experiment import paired_bootstrap_ci


class TestRLSchedulingEnv(unittest.TestCase):
    def test_reset_and_step_contract(self):
        env = RLSchedulingEnv([ProcessSpec("P1", 0, [2]), ProcessSpec("P2", 0, [1])])
        state = env.reset()
        self.assertIsInstance(state, tuple)
        next_state, reward, done, info = env.step(0)
        self.assertIsInstance(next_state, tuple)
        self.assertIsInstance(reward, float)
        self.assertIn("selected_pid", info)
        self.assertFalse(done)

    def test_episode_completes_and_metrics_are_consistent(self):
        env = RLSchedulingEnv([
            ProcessSpec("P1", 0, [2, 1], [2], process_type="io"),
            ProcessSpec("P2", 1, [3], priority=1),
        ])
        state = env.reset()
        done = False
        steps = 0
        while not done:
            state, reward, done, _ = env.step(4)  # shortest total remaining
            steps += 1
            self.assertLess(steps, 100)
        result = env.result()
        self.assertEqual(len(result.per_process), 2)
        self.assertEqual(sum(p["cpu_time"] for p in result.per_process), 6)
        self.assertEqual(result.metrics["makespan"], max(p["completion_time"] for p in result.per_process))
        self.assertEqual(result.decision_count, steps)

    def test_invalid_action_rejected(self):
        env = RLSchedulingEnv([ProcessSpec("P1", 0, [1])])
        with self.assertRaises(ValueError):
            env.step(N_ACTIONS)

    def test_invalid_epsilon_and_agent_action_rejected(self):
        agent = QLearningAgent(seed=1)
        with self.assertRaises(ValueError):
            agent.choose_action((0,), epsilon=1.1)
        with self.assertRaises(ValueError):
            agent.update((0,), True, 1.0, (1,), False)

    def test_agent_update_and_save_load(self):
        agent = QLearningAgent(seed=1)
        state = (1, 2, 3)
        next_state = (2, 1, 0)
        agent.update(state, 0, 1.0, next_state, False)
        self.assertNotEqual(agent.values(state)[0], 0.0)
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "q.json")
            agent.save(path)
            loaded = QLearningAgent.load(path, seed=2)
            self.assertEqual(loaded.values(state), agent.values(state))

    def test_action_names_are_unique(self):
        self.assertEqual(len(ACTION_NAMES), N_ACTIONS)
        self.assertEqual(len(set(ACTION_NAMES)), N_ACTIONS)

    def test_paired_bootstrap_interval_is_reproducible(self):
        differences = [-3.0, -2.0, -1.0, -4.0, -2.5]
        first = paired_bootstrap_ci(differences, seed=12, resamples=1000)
        second = paired_bootstrap_ci(differences, seed=12, resamples=1000)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], sum(differences) / len(differences))
        self.assertGreaterEqual(first[1], sum(differences) / len(differences))


if __name__ == "__main__":
    unittest.main()
