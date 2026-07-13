# Development Log

A running log of what's been built, key decisions, and what's next. Newest entry on top.

---

## 2026-07-12 — Core discrete-event engine (skeleton)

Commit: `53ebb13` — *"Added skeleton timeline and event! works fine, schedules a callable and runs it."*
(⚠️ needs a follow-up/amend — see **Housekeeping** below.)

### What was built

**`Event`** — `src/qetwork/events/event.py`
- A `@dataclass(slots=True)` representing **one deferred call scheduled at a virtual time**.
- Fields: `time: int` (picoseconds), `action: Callable`, `args: tuple`, `kwargs: dict`
  (via `field(default_factory=dict)` to avoid the shared-mutable-default trap), `priority: int`.
- `run()` executes the stored call: `action(*args, **kwargs)`.
- Deliberately has **no** ordering method and **no** cancellation/`valid` flag (see decisions).

**`Timeline`** — `src/qetwork/kernel/timeline.py`
- The discrete-event engine. Virtual clock in **integer picoseconds**.
- State: `time`, `stop_time` (default `math.inf`), `is_running`, `run_counter`,
  `queue` (a `heapq` list), `counter` (monotonic tie-breaker).
- `now()` → current virtual time.
- `schedule(action, *args, at, delay=0, priority=0, **kwargs)` → builds an `Event` at
  `time = at + delay`, pushes `(time, priority, counter, event)` onto the min-heap, bumps
  `counter`, returns the `Event`. `at` is a **mandatory keyword-only** absolute time;
  `delay` is an optional offset.
- `run()` → drains the heap earliest-first, advances the clock to each event's `time`,
  calls `event.run()` (which may schedule further events), tallies `run_counter`; stops
  when the queue empties or the next event's time exceeds `stop_time`.

### Key design decisions
- **Integer picoseconds** for time — no float drift, deterministic ordering.
- **Min-heap of `(time, priority, counter, event)` tuples.** The always-increasing `counter`
  makes every tuple unique → gives **FIFO** tie-breaking *and* guarantees two `Event` objects
  are never compared (they define no ordering).
- `priority` is an **`int`** (discrete ordering tiers), lower runs first.
- `schedule` uses a **mandatory absolute `at` + optional `delay`** — delaying from t=0 was
  deemed meaningless, so the anchor time is always explicit.
- **Direct callables** as the action IR; args kept **explicit** (not hidden in closures) to
  leave a seam for a future serializable / parallel kernel.
- **Cancellation + lazy deletion deferred** (not a rudimentary requirement).
- Clean-room **Pythonic** style (dataclasses, type hints).

### Verified (via `src/qetwork/tester.py` and one-liners)
- Schedules and runs a callable.
- Time-ordered execution — a later-scheduled but earlier-time event fires first.
- **Self-propagating events** — an event scheduling the next one (`tick` demo) runs to completion.

### Current file map
```
src/qetwork/
  events/event.py       # Event
  kernel/timeline.py    # Timeline
  tester.py             # scratch experiments
docs/
  timeline_ir.md        # kernel/timeline design doc
  dev_log.md            # this file
```

### Deferred / next
- **Entity** base class + two-phase startup (`add_entity`, `get_entity`, `Timeline.init()`),
  plus the `entities` registry (not yet in `__init__`).
- `step()` and `run_until(t)` loop controls.
- Extract an `EventQueue` (behind a `Protocol`) from the inline `heapq`.
- Reproducible RNG (`get_generator`).
- Quantum state manager hook.
- Event cancellation / rescheduling.

### Housekeeping (git)
- `src/qetwork/events/` (incl. `event.py`) is **untracked** — the committed `timeline.py`
  imports it, so the commit does not currently run. Needs to be added.
- Doc rename `timeline.md → timeline_ir.md` is uncommitted.
- Decide whether `tester.py` belongs in history.
