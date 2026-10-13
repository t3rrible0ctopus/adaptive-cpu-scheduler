# Simulator Validation Log — v1.1

## Validation performed
- FCFS textbook-style completion, waiting, turnaround, and response-time checks.
- Non-preemptive SJF behavior and shortest-job selection among simultaneously ready jobs.
- Round Robin with a known quantum-2 schedule.
- Preemptive priority behavior (lower numeric priority is higher priority).
- CPU idle before first arrival and CPU-utilization denominator.
- I/O blocking and return to the ready queue.
- Decision-overhead accounting is reported separately and does not alter simulated time.
- Invalid I/O burst shape and duplicate process IDs are rejected.

## Metric definitions
- Turnaround = completion time - arrival time.
- Response = first CPU start - arrival time.
- Waiting = total time spent in the ready queue, excluding I/O time.
- Throughput = completed processes / makespan.
- CPU utilization = CPU busy ticks / makespan.
- Context switches = transitions between different process IDs; idle dispatches and
  resuming the same PID after I/O are excluded by this simulator's convention.
- Fairness remains a service-rate proxy (CPU demand / turnaround); label it clearly and
  reconsider it before making final research claims.

## Known caveats
- Priority is preemptive; specify this choice in the methodology.
- SJF uses the next CPU burst as if known, making it an idealized baseline.
- Decision overhead is accounted separately, not simulated as CPU time.
- Workloads are synthetic; they do not establish real-system generalization.
