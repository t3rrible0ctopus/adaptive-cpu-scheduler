"""Single-core CPU scheduling simulator for the adaptive RL scheduler project.

Policies:
- FCFS: non-preemptive, first ready first served.
- SJF: non-preemptive, shortest next CPU burst first.
- RR: preemptive Round Robin with configurable time quantum.
- Priority: preemptive priority scheduling; lower numeric value means higher priority.

Processes can have multiple CPU bursts separated by I/O waits. All time is measured
in integer simulation ticks. No real sleeping or operating-system process scheduling
is performed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Callable, Iterable, Optional, Any
import math


@dataclass
class ProcessSpec:
    pid: str
    arrival_time: int
    cpu_bursts: list[int]
    io_bursts: list[int] = field(default_factory=list)
    priority: int = 0
    process_type: str = "cpu"

    def validate(self) -> None:
        if not isinstance(self.pid, str) or not self.pid:
            raise ValueError("pid must be non-empty")
        if isinstance(self.arrival_time, bool) or not isinstance(self.arrival_time, int) or self.arrival_time < 0:
            raise ValueError(f"{self.pid}: arrival_time must be >= 0")
        if (not isinstance(self.cpu_bursts, list) or not self.cpu_bursts
                or any(isinstance(b, bool) or not isinstance(b, int) or b <= 0
                       for b in self.cpu_bursts)):
            raise ValueError(f"{self.pid}: cpu_bursts must contain positive integers")
        if (not isinstance(self.io_bursts, list)
                or len(self.io_bursts) != len(self.cpu_bursts) - 1):
            raise ValueError(
                f"{self.pid}: expected exactly len(cpu_bursts)-1 I/O bursts"
            )
        if any(isinstance(b, bool) or not isinstance(b, int) or b < 0
               for b in self.io_bursts):
            raise ValueError(f"{self.pid}: I/O bursts must be >= 0")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ValueError(f"{self.pid}: priority must be an integer")
        if not isinstance(self.process_type, str) or not self.process_type:
            raise ValueError(f"{self.pid}: process_type must be a non-empty string")


@dataclass
class ProcessState:
    spec: ProcessSpec
    burst_index: int = 0
    remaining_in_burst: int = 0
    remaining_total_cpu: int = 0
    ready_since: Optional[int] = None
    first_start: Optional[int] = None
    completion_time: Optional[int] = None
    waiting_time: int = 0
    cpu_time: int = 0
    io_time: int = 0
    ready_wait_accumulated: int = 0
    state: str = "new"  # new, ready, running, io, done
    enqueue_order: int = 0

    def __post_init__(self):
        self.remaining_in_burst = self.spec.cpu_bursts[0]
        self.remaining_total_cpu = sum(self.spec.cpu_bursts)


@dataclass
class SimulationResult:
    policy: str
    metrics: dict[str, float]
    per_process: list[dict[str, Any]]
    timeline: list[dict[str, Any]]
    decision_count: int
    simulated_decision_overhead: float

    def summary(self) -> dict[str, Any]:
        return {"policy": self.policy, **self.metrics}


class CPUSchedulerSimulator:
    """Discrete-time, single-core scheduling simulator.

    decision_overhead_per_decision is accounting overhead, not inserted into the
    scheduling clock. It is reported separately so baseline policy comparisons stay
    interpretable. A later version may model overhead as consumed CPU time.
    """

    VALID_POLICIES = {"FCFS", "SJF", "RR", "PRIORITY"}

    def __init__(
        self,
        processes: Iterable[ProcessSpec],
        policy: str,
        quantum: int = 3,
        decision_overhead_per_decision: float = 0.0,
        aging_interval: Optional[int] = None,
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
        policy = policy.upper()
        if policy not in self.VALID_POLICIES:
            raise ValueError(f"policy must be one of {sorted(self.VALID_POLICIES)}")
        if quantum <= 0:
            raise ValueError("quantum must be > 0")
        if decision_overhead_per_decision < 0:
            raise ValueError("decision overhead must be >= 0")
        if aging_interval is not None and aging_interval <= 0:
            raise ValueError("aging_interval must be > 0")
        self.policy = policy
        self.quantum = quantum
        self.overhead = float(decision_overhead_per_decision)
        self.aging_interval = aging_interval

    def run(self) -> SimulationResult:
        states = {p.pid: ProcessState(p) for p in self.specs}
        ready: list[ProcessState] = []
        io_events: list[tuple[int, str]] = []
        timeline: list[dict[str, Any]] = []
        t = 0
        enqueue_counter = 0
        current: Optional[ProcessState] = None
        rr_slice = 0
        last_pid: Optional[str] = None
        context_switches = 0
        decisions = 0
        busy_ticks = 0
        completed = 0
        total_cpu = sum(sum(p.cpu_bursts) for p in self.specs)
        last_event_time = 0

        def add_ready(ps: ProcessState, now: int):
            nonlocal enqueue_counter
            ps.state = "ready"
            ps.ready_since = now
            ps.enqueue_order = enqueue_counter
            enqueue_counter += 1
            ready.append(ps)

        def record_segment(start, end, pid, kind):
            if end > start:
                if timeline and timeline[-1]["pid"] == pid and timeline[-1]["kind"] == kind and timeline[-1]["end"] == start:
                    timeline[-1]["end"] = end
                else:
                    timeline.append({"start": start, "end": end, "pid": pid, "kind": kind})

        def release_events(now: int):
            # Arrivals and I/O completions are enqueued deterministically by time then PID.
            arrivals = [s for s in states.values()
                        if s.state == "new" and s.spec.arrival_time <= now]
            for s in sorted(arrivals, key=lambda x: (x.spec.arrival_time, x.spec.pid)):
                add_ready(s, now)
            done_io = [(when, pid) for when, pid in io_events if when <= now]
            for when, pid in sorted(done_io, key=lambda x: (x[0], x[1])):
                s = states[pid]
                if s.state == "io":
                    s.io_time += when - getattr(s, "_io_start", when)
                    add_ready(s, now)
            if done_io:
                io_events[:] = [event for event in io_events if event[0] > now]

        def effective_priority(s: ProcessState, now: int) -> float:
            if self.aging_interval is None:
                return s.spec.priority
            waited = max(0, now - (s.ready_since if s.ready_since is not None else now))
            return s.spec.priority - (waited // self.aging_interval)

        def choose_ready(now: int) -> Optional[ProcessState]:
            if not ready:
                return None
            if self.policy in ("FCFS", "RR"):
                # RR uses FIFO order when a time slice expires.
                chosen = min(ready, key=lambda s: (s.enqueue_order, s.spec.pid))
            elif self.policy == "SJF":
                chosen = min(ready, key=lambda s: (s.remaining_in_burst, s.enqueue_order, s.spec.pid))
            else:
                chosen = min(ready, key=lambda s: (effective_priority(s, now), s.enqueue_order, s.spec.pid))
            ready.remove(chosen)
            if chosen.ready_since is not None:
                chosen.waiting_time += now - chosen.ready_since
                chosen.ready_wait_accumulated += now - chosen.ready_since
                chosen.ready_since = None
            chosen.state = "running"
            return chosen

        # Advance until every process has completed all CPU bursts.
        while completed < len(states):
            release_events(t)

            # Preemptive priority policy: a newly ready higher-priority process may preempt.
            if current is not None and self.policy == "PRIORITY" and ready:
                best = min(ready, key=lambda s: (effective_priority(s, t), s.enqueue_order, s.spec.pid))
                if effective_priority(best, t) < effective_priority(current, t):
                    current.state = "ready"
                    current.ready_since = t
                    current.enqueue_order = enqueue_counter
                    enqueue_counter += 1
                    ready.append(current)
                    current = None

            if current is None:
                current = choose_ready(t)
                if current is not None:
                    decisions += 1
                    if last_pid is not None and last_pid != current.spec.pid:
                        context_switches += 1
                    if current.first_start is None:
                        current.first_start = t
                    rr_slice = 0
                    last_pid = current.spec.pid

            if current is None:
                # CPU idle: jump to the next arrival or I/O completion.
                future = [s.spec.arrival_time for s in states.values() if s.state == "new"]
                future += [when for when, pid in io_events if states[pid].state == "io"]
                if not future:
                    raise RuntimeError("Simulation deadlock: unfinished processes but no future events")
                next_t = min(future)
                if next_t <= t:
                    next_t = t + 1
                record_segment(t, next_t, None, "idle")
                # Idle breaks CPU ownership; dispatch after idle is not counted as
                # a process-to-process context switch under this project's metric.
                last_pid = None
                t = next_t
                continue

            # Execute one CPU tick.
            record_segment(t, t + 1, current.spec.pid, "cpu")
            current.remaining_in_burst -= 1
            current.remaining_total_cpu -= 1
            current.cpu_time += 1
            busy_ticks += 1
            rr_slice += 1
            t += 1

            # Finish a CPU burst.
            if current.remaining_in_burst == 0:
                if current.burst_index < len(current.spec.io_bursts):
                    io_duration = current.spec.io_bursts[current.burst_index]
                    current.burst_index += 1
                    current.state = "io"
                    setattr(current, "_io_start", t)
                    io_events.append((t + io_duration, current.spec.pid))
                    current.remaining_in_burst = current.spec.cpu_bursts[current.burst_index]
                    current = None
                else:
                    current.state = "done"
                    current.completion_time = t
                    completed += 1
                    current = None
                continue

            # Round Robin quantum expiry; arrivals at the boundary are released next loop.
            if self.policy == "RR" and rr_slice >= self.quantum:
                current.state = "ready"
                current.ready_since = t
                current.enqueue_order = enqueue_counter
                enqueue_counter += 1
                ready.append(current)
                current = None

        makespan = max((s.completion_time or 0) for s in states.values())
        per_process = []
        for s in sorted(states.values(), key=lambda x: x.spec.pid):
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
        # Jain's index is computed over per-process CPU service divided by turnaround,
        # a bounded service-rate proxy. It is documented as a proxy, not a universal
        # fairness definition for CPU scheduling.
        service_rates = [
            (p["total_cpu_burst"] / p["turnaround_time"]) if p["turnaround_time"] > 0 else 0.0
            for p in per_process
        ]
        denom = len(service_rates) * sum(x * x for x in service_rates)
        jain = (sum(service_rates) ** 2 / denom) if denom > 0 else 1.0
        total_elapsed = max(1, makespan)
        metrics = {
            "average_waiting_time": sum(waits) / len(waits),
            "average_turnaround_time": sum(turns) / len(turns),
            "average_response_time": sum(responses) / len(responses),
            "throughput_processes_per_tick": len(per_process) / total_elapsed,
            "cpu_utilization": busy_ticks / total_elapsed,
            # Transitions between different process IDs only. Idle dispatches and
            # resuming the same PID after I/O are excluded by this convention.
            "context_switches": context_switches,
            "max_waiting_time": max(waits, default=0),
            "jain_fairness_index_service_rate_proxy": jain,
            "makespan": makespan,
            "total_cpu_work": total_cpu,
            "scheduling_decisions": decisions,
            "decision_overhead_total": decisions * self.overhead,
            "decision_overhead_per_decision": self.overhead,
        }
        return SimulationResult(
            policy=self.policy,
            metrics=metrics,
            per_process=per_process,
            timeline=timeline,
            decision_count=decisions,
            simulated_decision_overhead=decisions * self.overhead,
        )


def simulate(processes: Iterable[ProcessSpec], policy: str, **kwargs) -> SimulationResult:
    """Convenience function."""
    return CPUSchedulerSimulator(processes, policy, **kwargs).run()
