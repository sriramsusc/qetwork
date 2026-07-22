"""Same-tick event ordering: lower runs first.

The heap orders (time, priority, counter), so priority ONLY breaks ties between
events at the same picosecond tick; counter then gives FIFO within a tier.

Tick pipeline: control -> physical arrivals -> state operations -> protocol
decisions -> metrics. A protocol event landing on the same tick as an arrival
must observe that arrival's effects (a retry check must see the herald that
landed this tick), so PROTOCOL sits after ARRIVAL and OPERATION.
"""

CONTROL   = -100   # simulation control: kickoffs, mode switches        (reserved, unused)
ARRIVAL   = 0      # photons at ports, messages at nodes, detector clicks
OPERATION = 10     # scheduled state changes: gates/noise as events     (reserved — noise is
                   #   currently synchronous inside node verbs)
PROTOCOL  = 20     # agent reactions to this tick's arrivals: retries, byproduct sends, kickoffs
METRIC    = 100    # observers reading settled state: fidelity probes, loggers
