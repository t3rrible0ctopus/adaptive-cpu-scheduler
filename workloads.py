"""Reproducible workload generation for CPU scheduling experiments."""
from __future__ import annotations
from scheduler_sim import ProcessSpec
import random


LOAD_LEVELS = {
    # Arrival interval and approximate offered CPU load. The generator is intended
    # for controlled comparisons, not as a calibrated trace model.
    "light": {"arrival_gap": (5, 9), "process_count": 12},
    "medium": {"arrival_gap": (2, 5), "process_count": 20},
    "heavy": {"arrival_gap": (1, 3), "process_count": 30},
}


def generate_workload(kind: str, load: str, seed: int = 42) -> list[ProcessSpec]:
    """Create CPU-bound, I/O-bound or mixed workloads.

    CPU-bound: one relatively long CPU burst.
    I/O-bound: several short CPU bursts separated by I/O waits.
    Mixed: mixture of both patterns.

    A fixed seed produces the same workload, enabling paired comparisons.
    """
    kind = kind.lower()
    load = load.lower()
    if kind not in {"cpu", "io", "mixed"}:
        raise ValueError("kind must be 'cpu', 'io', or 'mixed'")
    if load not in LOAD_LEVELS:
        raise ValueError("load must be 'light', 'medium', or 'heavy'")

    rng = random.Random(seed)
    config = LOAD_LEVELS[load]
    count = config["process_count"]
    gap_lo, gap_hi = config["arrival_gap"]
    t = 0
    processes = []

    for i in range(count):
        if i:
            t += rng.randint(gap_lo, gap_hi)
        chosen_type = kind
        if kind == "mixed":
            chosen_type = "cpu" if rng.random() < 0.5 else "io"

        priority = rng.randint(0, 4)
        if chosen_type == "cpu":
            cpu_bursts = [rng.randint(8, 24)]
            io_bursts = []
        else:
            n_bursts = rng.randint(3, 6)
            cpu_bursts = [rng.randint(1, 4) for _ in range(n_bursts)]
            io_bursts = [rng.randint(3, 10) for _ in range(n_bursts - 1)]

        processes.append(ProcessSpec(
            pid=f"P{i+1:02d}",
            arrival_time=t,
            cpu_bursts=cpu_bursts,
            io_bursts=io_bursts,
            priority=priority,
            process_type=chosen_type,
        ))
    return processes


def describe_workload(processes: list[ProcessSpec]) -> dict:
    total_cpu = sum(sum(p.cpu_bursts) for p in processes)
    total_io = sum(sum(p.io_bursts) for p in processes)
    return {
        "process_count": len(processes),
        "total_cpu_burst_ticks": total_cpu,
        "total_io_wait_ticks": total_io,
        "cpu_bound_count": sum(p.process_type == "cpu" for p in processes),
        "io_bound_count": sum(p.process_type == "io" for p in processes),
        "first_arrival": min(p.arrival_time for p in processes),
        "last_arrival": max(p.arrival_time for p in processes),
    }
