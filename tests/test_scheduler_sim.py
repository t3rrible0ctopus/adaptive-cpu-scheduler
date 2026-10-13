import unittest
from scheduler_sim import ProcessSpec, simulate


class TestSchedulerSimulator(unittest.TestCase):
    def test_fcfs_known_result(self):
        jobs = [ProcessSpec("P1", 0, [5]), ProcessSpec("P2", 1, [3])]
        result = simulate(jobs, "FCFS")
        rows = {p["pid"]: p for p in result.per_process}
        self.assertEqual(rows["P1"]["completion_time"], 5)
        self.assertEqual(rows["P2"]["completion_time"], 8)
        self.assertEqual(rows["P2"]["waiting_time"], 4)
        self.assertEqual(rows["P2"]["turnaround_time"], 7)
        self.assertEqual(rows["P2"]["response_time"], 4)
        self.assertEqual(result.metrics["makespan"], 8)
        self.assertAlmostEqual(result.metrics["cpu_utilization"], 1.0)

    def test_sjf_is_non_preemptive(self):
        jobs = [ProcessSpec("P1", 0, [6]), ProcessSpec("P2", 1, [2])]
        result = simulate(jobs, "SJF")
        rows = {p["pid"]: p for p in result.per_process}
        self.assertEqual(rows["P1"]["completion_time"], 6)
        self.assertEqual(rows["P2"]["completion_time"], 8)
        self.assertEqual(rows["P2"]["waiting_time"], 5)
        self.assertEqual(rows["P2"]["response_time"], 5)

    def test_sjf_selects_shortest_when_both_ready(self):
        jobs = [ProcessSpec("P1", 0, [6]), ProcessSpec("P2", 0, [2])]
        result = simulate(jobs, "SJF")
        rows = {p["pid"]: p for p in result.per_process}
        self.assertEqual(rows["P2"]["completion_time"], 2)
        self.assertEqual(rows["P1"]["completion_time"], 8)

    def test_round_robin_known_result(self):
        jobs = [ProcessSpec("P1", 0, [4]), ProcessSpec("P2", 0, [3])]
        result = simulate(jobs, "RR", quantum=2)
        rows = {p["pid"]: p for p in result.per_process}
        self.assertEqual(rows["P1"]["completion_time"], 6)
        self.assertEqual(rows["P2"]["completion_time"], 7)
        self.assertEqual(rows["P1"]["waiting_time"], 2)
        self.assertEqual(rows["P2"]["waiting_time"], 4)
        self.assertEqual(result.metrics["context_switches"], 3)

    def test_priority_preempts_lower_priority(self):
        jobs = [ProcessSpec("P1", 0, [5], priority=3), ProcessSpec("P2", 2, [2], priority=0)]
        result = simulate(jobs, "PRIORITY")
        rows = {p["pid"]: p for p in result.per_process}
        self.assertEqual(rows["P2"]["response_time"], 0)
        self.assertEqual(rows["P2"]["completion_time"], 4)
        self.assertEqual(rows["P1"]["completion_time"], 7)

    def test_cpu_idle_before_first_arrival(self):
        result = simulate([ProcessSpec("P1", 5, [3])], "FCFS")
        self.assertEqual(result.metrics["makespan"], 8)
        self.assertAlmostEqual(result.metrics["cpu_utilization"], 3 / 8)
        self.assertEqual(result.per_process[0]["response_time"], 0)
        self.assertEqual(result.timeline[0], {"start": 0, "end": 5, "pid": None, "kind": "idle"})

    def test_io_process_blocks_and_returns(self):
        jobs = [ProcessSpec("P1", 0, [2, 2], [3]), ProcessSpec("P2", 0, [5])]
        result = simulate(jobs, "FCFS")
        rows = {p["pid"]: p for p in result.per_process}
        self.assertEqual(rows["P1"]["cpu_time"], 4)
        self.assertEqual(rows["P1"]["io_time"], 3)
        self.assertGreater(rows["P1"]["turnaround_time"], rows["P1"]["cpu_time"])

    def test_decision_overhead_is_accounted_separately(self):
        result = simulate([ProcessSpec("P1", 0, [3])], "FCFS", decision_overhead_per_decision=0.25)
        self.assertEqual(result.decision_count, 1)
        self.assertAlmostEqual(result.simulated_decision_overhead, 0.25)
        self.assertEqual(result.metrics["makespan"], 3)

    def test_invalid_io_shape_rejected(self):
        with self.assertRaises(ValueError):
            ProcessSpec("P1", 0, [2, 3], []).validate()

    def test_duplicate_pids_rejected(self):
        with self.assertRaises(ValueError):
            simulate([ProcessSpec("P1", 0, [1]), ProcessSpec("P1", 1, [1])], "FCFS")


if __name__ == "__main__":
    unittest.main()
