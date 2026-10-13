"""A small, tabular-Q-learning environment for adaptive CPU scheduling.

The environment is a single-core, discrete-time simulator. At each CPU tick the
agent chooses a dispatch rule, and that rule selects the ready process to run for
one tick. This indirect action space keeps Q-learning tractable while still making
the agent responsible for the next process selection.

This is a research prototype, not a real kernel scheduler.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Optional

from scheduler_sim import ProcessSpec, ProcessState, SimulationResult


ACTION_NAMES = (
    "OLDEST_READY",
    "SHORTEST_CURRENT_BURST",
    "HIGHEST_PRIORITY",
    "LONGEST_WAITING",
    "SHORTEST_TOTAL_REMAINING",
    "CONTINUE_LAST_PROCESS",
)
N_ACTIONS = len(ACTION_NAMES)


def _bin(value: float, edges: tuple[float, ...]) -> int:
    for i, edge in enumerate(edges):
        if value <= edge:
            return i
    return len(edges)


class RLSchedulingEnv:
    """Gym-like reset/step interface without an external Gym dependency.

    Observation/state is a hashable tuple of discretized queue statistics.
    Actions select a dispatch heuristic; the heuristic selects an eligible ready
    process. Every action advances exactly one CPU tick, allowing preemption at
    each tick. When the ready queue empties, the environment advances through idle
    time to the next arrival or I/O completion before returning the next state.
    """

    def __init__(
        self,
        processes: Iterable[ProcessSpec],
        decision_overhead_per_decision: float = 0.0,
        switch_penalty: float = 0.15,
        starvation_threshold: int = 10,
        starvation_penalty: float = 0.75,
    ):
        self.specs = [ProcessSpec(p.pid, p.arrival_time, list(p.cpu_bursts),
                                  list(p.io_bursts), p.priority, p.process_type)
                      for p in processes]
        if not self.specs:
            raise ValueError("At least one process is required")
        for p in self.specs:
            p.validate()
        if len({p.pid for p in self.specs}) != len(self.specs):
            raise ValueError("Process IDs must be unique")
        if decision_overhead_per_decision < 0:
            raise ValueError("decision_overhead_per_decision must be >= 0")
        if switch_penalty < 0 or starvation_penalty < 0:
            raise ValueError("Reward penalties must be >= 0")
        if starvation_threshold <= 0:
            raise ValueError("starvation_threshold must be > 0")
        self.overhead = float(decision_overhead_per_decision)
        self.switch_penalty = float(switch_penalty)
        self.starvation_threshold = int(starvation_threshold)
        self.starvation_penalty = float(starvation_penalty)
        self.reset()

    def reset(self) -> tuple:
        self.states = {p.pid: ProcessState(p) for p in self.specs}
        self.ready: list[ProcessState] = []
        self.io_events: list[tuple[int, str]] = []
        self.timeline: list[dict[str, Any]] = []
        self.t = 0
        self.enqueue_counter = 0
        self.last_pid: Optional[str] = None
        self.context_switches = 0
        self.decisions = 0
        self.busy_ticks = 0
        self.completed = 0
        self.total_cpu = sum(sum(p.cpu_bursts) for p in self.specs)
        self._release_events()
        self._advance_to_ready_or_done()
        return self._observation()

    def _add_ready(self, ps: ProcessState, now: int) -> None:
        ps.state = "ready"
        ps.ready_since = now
        ps.enqueue_order = self.enqueue_counter
        self.enqueue_counter += 1
        self.ready.append(ps)

    def _release_events(self) -> None:
        arrivals = [s for s in self.states.values()
                    if s.state == "new" and s.spec.arrival_time <= self.t]
        for s in sorted(arrivals, key=lambda x: (x.spec.arrival_time, x.spec.pid)):
            self._add_ready(s, self.t)
        done_io = sorted((when, pid) for when, pid in self.io_events if when <= self.t)
        for when, pid in done_io:
            s = self.states[pid]
            if s.state == "io":
                s.io_time += when - getattr(s, "_io_start", when)
                self._add_ready(s, self.t)
        if done_io:
            self.io_events = [event for event in self.io_events if event[0] > self.t]

    def _record_segment(self, start: int, end: int, pid: Optional[str], kind: str) -> None:
        if end <= start:
            return
        if (self.timeline and self.timeline[-1]["pid"] == pid
                and self.timeline[-1]["kind"] == kind
                and self.timeline[-1]["end"] == start):
            self.timeline[-1]["end"] = end
        else:
            self.timeline.append({"start": start, "end": end, "pid": pid, "kind": kind})

    def _advance_to_ready_or_done(self) -> None:
        while not self.ready and self.completed < len(self.states):
            self._release_events()
            if self.ready or self.completed == len(self.states):
                return
            future = [s.spec.arrival_time for s in self.states.values() if s.state == "new"]
            future.extend(when for when, pid in self.io_events if self.states[pid].state == "io")
            if not future:
                raise RuntimeError("Environment deadlock: no ready process or future event")
            next_t = min(future)
            if next_t > self.t:
                self._record_segment(self.t, next_t, None, "idle")
                # Idle breaks CPU ownership for context-switch accounting.
                self.last_pid = None
                self.t = next_t
            self._release_events()

    def _waiting_now(self, ps: ProcessState) -> int:
        return max(0, self.t - (ps.ready_since if ps.ready_since is not None else self.t))

    def _observation(self) -> tuple:
        if not self.ready:
            return (0, 0, 0, 0, 0, 0, 0, 0, 0)
        ready = sorted(self.ready, key=lambda s: (s.spec.pid, s.enqueue_order))
        waits = [self._waiting_now(s) for s in ready]
        bursts = [s.remaining_in_burst for s in ready]
        io_fraction = sum(s.spec.process_type == "io" for s in ready) / len(ready)
        high_priority_fraction = sum(s.spec.priority <= 1 for s in ready) / len(ready)
        blocked_count = sum(s.state == "io" for s in self.states.values())
        last_is_ready = int(any(s.spec.pid == self.last_pid for s in ready))
        return (
            _bin(len(ready), (1, 2, 4, 8)),
            _bin(sum(waits) / len(waits), (0, 3, 7, 15)),
            _bin(max(waits), (0, 3, 7, 15)),
            _bin(min(bursts), (2, 5, 10)),
            _bin(sum(bursts) / len(bursts), (3, 7, 15)),
            _bin(io_fraction, (0, 0.33, 0.66)),
            _bin(high_priority_fraction, (0.25, 0.5)),
            _bin(blocked_count, (0, 1, 4)),
            last_is_ready,
        )

    def _choose_process(self, action: int) -> ProcessState:
        if not isinstance(action, int) or not 0 <= action < N_ACTIONS:
            raise ValueError(f"action must be an integer from 0 to {N_ACTIONS - 1}")
        if not self.ready:
            raise RuntimeError("Cannot select a process when ready queue is empty")
        ready = list(self.ready)
        if action == 0:  # oldest ready process; preemptive at every tick boundary
            chosen = min(ready, key=lambda s: (s.ready_since, s.enqueue_order, s.spec.pid))
        elif action == 1:  # shortest current CPU burst; preemptive at every tick boundary
            chosen = min(ready, key=lambda s: (s.remaining_in_burst, s.enqueue_order, s.spec.pid))
        elif action == 2:  # highest priority; lower numeric value wins
            chosen = min(ready, key=lambda s: (s.spec.priority, s.enqueue_order, s.spec.pid))
        elif action == 3:  # longest waiting
            chosen = max(ready, key=lambda s: (self._waiting_now(s), -s.enqueue_order, s.spec.pid))
        elif action == 4:  # shortest total remaining CPU work
            chosen = min(ready, key=lambda s: (s.remaining_total_cpu, s.enqueue_order, s.spec.pid))
        else:  # continue the previous process if it is still ready, otherwise FCFS
            chosen = next((s for s in ready if s.spec.pid == self.last_pid), None)
            if chosen is None:
                chosen = min(ready, key=lambda s: (s.ready_since, s.enqueue_order, s.spec.pid))
        self.ready.remove(chosen)
        return chosen

    def step(self, action: int) -> tuple[tuple, float, bool, dict[str, Any]]:
        if self.completed >= len(self.states):
            raise RuntimeError("Episode is already complete; call reset()")
        self._release_events()
        self._advance_to_ready_or_done()
        if self.completed >= len(self.states):
            return self._observation(), 0.0, True, {"time": self.t}

        waiting_count_during_tick = max(0, len(self.ready) - 1)
        max_wait = max((self._waiting_now(s) for s in self.ready), default=0)
        chosen = self._choose_process(action)
        if chosen.ready_since is not None:
            chosen.waiting_time += self.t - chosen.ready_since
            chosen.ready_wait_accumulated += self.t - chosen.ready_since
            chosen.ready_since = None
        chosen.state = "running"
        if chosen.first_start is None:
            chosen.first_start = self.t

        switched = self.last_pid is not None and self.last_pid != chosen.spec.pid
        if switched:
            self.context_switches += 1
        self.last_pid = chosen.spec.pid
        self.decisions += 1

        self._record_segment(self.t, self.t + 1, chosen.spec.pid, "cpu")
        chosen.remaining_in_burst -= 1
        chosen.remaining_total_cpu -= 1
        chosen.cpu_time += 1
        self.busy_ticks += 1
        self.t += 1

        if chosen.remaining_in_burst == 0:
            if chosen.burst_index < len(chosen.spec.io_bursts):
                io_duration = chosen.spec.io_bursts[chosen.burst_index]
                chosen.burst_index += 1
                chosen.state = "io"
                setattr(chosen, "_io_start", self.t)
                self.io_events.append((self.t + io_duration, chosen.spec.pid))
                chosen.remaining_in_burst = chosen.spec.cpu_bursts[chosen.burst_index]
            else:
                chosen.state = "done"
                chosen.completion_time = self.t
                self.completed += 1
        else:
            # Requeue after one tick. This makes the learned policy preemptive at
            # tick boundaries, with new arrivals released immediately afterwards.
            self._add_ready(chosen, self.t)

        self._release_events()
        self._advance_to_ready_or_done()
        starvation_count = sum(
            1 for s in self.ready if self._waiting_now(s) >= self.starvation_threshold
        )
        reward = (
            -float(waiting_count_during_tick)
            - (self.switch_penalty if switched else 0.0)
            - self.starvation_penalty * starvation_count
        )
        done = self.completed == len(self.states)
        info = {
            "time": self.t,
            "selected_pid": chosen.spec.pid,
            "action_name": ACTION_NAMES[action],
            "context_switch": switched,
            "waiting_processes_during_tick": waiting_count_during_tick,
            "completed_processes": self.completed,
        }
        return self._observation(), reward, done, info

    def result(self) -> SimulationResult:
        if self.completed != len(self.states):
            raise RuntimeError("Episode is not complete")
        per_process = []
        for s in sorted(self.states.values(), key=lambda x: x.spec.pid):
            turnaround = s.completion_time - s.spec.arrival_time
            response = s.first_start - s.spec.arrival_time
            per_process.append({
                "pid": s.spec.pid,
                "type": s.spec.process_type,
                "arrival_time": s.spec.arrival_time,
                "priority": s.spec.priority,
                "total_cpu_burst": sum(s.spec.cpu_bursts),
                "completion_time": s.completion_time,
                "waiting_time": s.waiting_time,
                "turnaround_time": turnaround,
                "response_time": response,
                "cpu_time": s.cpu_time,
                "io_time": s.io_time,
            })
        waits = [p["waiting_time"] for p in per_process]
        turns = [p["turnaround_time"] for p in per_process]
        responses = [p["response_time"] for p in per_process]
        rates = [(p["total_cpu_burst"] / p["turnaround_time"])
                 if p["turnaround_time"] > 0 else 0.0 for p in per_process]
        denom = len(rates) * sum(x * x for x in rates)
        jain = (sum(rates) ** 2 / denom) if denom > 0 else 1.0
        makespan = max((s.completion_time or 0) for s in self.states.values())
        elapsed = max(1, makespan)
        metrics = {
            "average_waiting_time": sum(waits) / len(waits),
            "average_turnaround_time": sum(turns) / len(turns),
            "average_response_time": sum(responses) / len(responses),
            "throughput_processes_per_tick": len(per_process) / elapsed,
            "cpu_utilization": self.busy_ticks / elapsed,
            "context_switches": self.context_switches,
            "max_waiting_time": max(waits, default=0),
            "jain_fairness_index_service_rate_proxy": jain,
            "makespan": makespan,
            "total_cpu_work": self.total_cpu,
            "scheduling_decisions": self.decisions,
            "decision_overhead_total": self.decisions * self.overhead,
            "decision_overhead_per_decision": self.overhead,
        }
        return SimulationResult(
            policy="QLEARNING",
            metrics=metrics,
            per_process=per_process,
            timeline=self.timeline,
            decision_count=self.decisions,
            simulated_decision_overhead=self.decisions * self.overhead,
        )
