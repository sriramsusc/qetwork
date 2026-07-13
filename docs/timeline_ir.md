# Timeline — Design

> **Scope of this document:** the **Timeline** is the focus. It is the discrete-event
> engine at the core of the quantum network simulator. Supporting components (Event,
> EventQueue, Entity) are described only as far as the Timeline needs them; each will
> get its own detailed design later.

## Decisions locked in

| Question | Decision |
|---|---|
| Parallel / distributed | **Single-process now**, but design the IR and seams so a parallel kernel drops in later |
| Action representation (IR) | **Direct callables** — `action` + explicit `args`/`kwargs` (not opaque closures) |
| API fidelity | **Clean-room Pythonic** — dataclasses, type hints, our own naming |
| Time unit | **Integer picoseconds** (no float clock) |
| Ordering | `(time, priority, seq)` — lower `priority` first, `seq` = FIFO tiebreak |
| Default priority | `0` (lower runs first), ties broken FIFO by insertion order |
| Cancellation | **Deferred** — not a rudimentary requirement (see "Deferred") |

## The computational model

A discrete-event simulator is a **fixed-point of scheduling**. The only primitive is:

> run action *A* at virtual time *T*.

An action mutates entity state and may `schedule` further actions. The Timeline
repeatedly pops the earliest pending event and runs it, advancing a **virtual clock**
(never wall-clock). Everything the network does — photon emission, channel delay,
Bell-state measurement, memory decoherence, classical messages — is expressed as
events on this one queue.

```
schedule(A@T) ──► [ EventQueue ] ──► pop earliest ──► advance clock ──► run A
      ▲                                                                   │
      └──────────────────  A may schedule more events  ◄─────────────────┘
```

---

## Timeline (the focus)

### State

```
time: int                 # current virtual time, picoseconds
events: EventQueue        # pending events (binary min-heap)
entities: dict[str, Entity]   # named registry = future cross-process address space
stop_time: int            # terminate when next event exceeds this (default: inf)
is_running: bool
rng                       # seedable RNG for reproducibility
schedule_counter: int     # stats: total events scheduled
run_counter: int          # stats: total events executed
# reserved, unimplemented:
# quantum_manager         # quantum state store/evolution (future)
```

### API

```python
Timeline(stop_time=inf, seed=None)

now() -> int                       # current virtual time

# entity registry
add_entity(entity)
get_entity(name) -> Entity
remove_entity(name)

# scheduling — steer toward schedule(entity.method, *args) (see "seams")
schedule(action, *args, delay=..., at=..., priority=0, **kwargs) -> Event
    # exactly one of delay (relative to now) / at (absolute); returns the Event
    # (useful for tests / introspection)

# lifecycle
init()          # call entity.init() for every registered entity
run()           # main loop until stop_time or the queue empties
run_until(t)    # bounded run; also the lookahead primitive for parallel sync
step()          # pop + run exactly one valid event
stop()          # request termination
```

### Lifecycle

Two phases, always in this order:

1. **Construct** entities — each auto-registers with the Timeline in its `__init__`.
2. **`init()`** — Timeline calls every `entity.init()`, which seeds the initial events.
3. **`run()`** — the event loop drains the queue, advancing the clock.

### Run loop

```
is_running = True
while events and not stopped:
    e = events.pop()
    if e.time > stop_time:  break             # (optionally push e back)
    time = e.time                             # advance virtual clock
    e.run()                                   # action(*args, **kwargs); may schedule more
    run_counter += 1
is_running = False
```

Key invariants:
- `time` is **monotonically non-decreasing**; an action must never schedule into the past.
- The clock only moves when an event is executed — no wall-clock, no fixed timestep.

---

## Supporting components (brief — detailed later)

These exist to serve the Timeline; full designs come in their own docs.

**Event** — one scheduled action.
```
time: int; action: Callable; args: tuple; kwargs: dict
priority: float = 0; seq: int (assigned by queue)
run()    -> action(*args, **kwargs)
# sort key = (time, priority, seq); never compares `action`
```

**EventQueue** — pending events, behind a `Protocol`/trait so a different queue can replace it.
```
push(event)  # assigns monotonic seq
pop() -> Event; peek() -> Event | None; __len__; __bool__
```
Backed by a **binary min-heap** over a flat array. With cancellation deferred, there is
no `valid` flag and no lazy deletion — just insert + extract-min. See
"Event queue data structure" for the Rust-port rationale and upgrade path.

**Entity** — abstract base for every stateful simulation object: nodes (repeaters,
end nodes), hardware (quantum memories, photon sources, detectors, BSM units), and
channels. The Timeline is a pure scheduler and holds **no** domain state; the entities
hold it, and events mutate it.
```
__init__(name, timeline)   # auto-registers via timeline.add_entity(self)
name; timeline; owner
init()            # abstract — seeds initial events (two-phase startup)
get_generator()   # reproducible RNG stream from the timeline
```
Why it exists: (1) **state ownership** — scheduler vs. state separation; (2) a unique
**name** = stable lookup handle and the future cross-process address; (3) the **`init()`**
phase lets the topology be fully wired before any event fires; (4) **reproducible RNG**
plumbing. Think Timeline = OS scheduler, Entity = process.

---

## Event queue data structure

With cancellation deferred, the pending-event set needs only **insert** and
**extract-min** — a plain priority queue.

**Choice: binary min-heap over a flat array.** `heapq` in Python; `BinaryHeap<Reverse<Event>>`
or a hand-rolled 4-ary heap over a `Vec<Event>` in Rust. Why this is the right default,
especially for a future Rust port:

- **Cache locality** — a heap is a contiguous array, no pointer chasing. This is where Rust
  wins: `Vec<Event>` is flat and owned, no `Rc`/`RefCell`. Pointer-based heaps (Fibonacci,
  pairing, splay tree) have nicer asymptotics on paper but worse real-world constants *and*
  fight the borrow checker — avoid them.
- **4-ary / 8-ary heap** — a cheap constant-factor win over a binary heap: shallower tree,
  fewer cache misses on sift-down. Trivial over a `Vec`; keep in mind for the port.
- **Simplicity** — O(log n) insert / extract-min, trivially correct, deterministic.

**Upgrade path (later, only if a profile demands it).** DES-specialised structures beat a
heap only at very large pending-event populations:

- **Calendar queue** — time-bucketed; ~O(1) amortised *if* event times are roughly uniform.
  Degrades under skew; needs periodic resizing.
- **Ladder queue** — multi-tier calendar queue that handles skew; state-of-the-art for
  large-scale / parallel DES (e.g. ROSS). More complex.

Both are premature now. Keeping the queue behind a `Protocol`/trait lets us swap one in
without touching the Timeline. **Recommendation: binary min-heap now; revisit only if the
event queue shows up in a profile.**

## The callable ↔ parallel tension (and its resolution)

Direct callables and future serialization mildly conflict: a `lambda`/closure cannot
cross a process boundary, but a **bound method of a named entity + explicit args** can.
So the Event IR stores the action *decomposed*, not as an opaque closure:

```
Event.action = entity.receive     # bound method: carries __self__ (named entity) + __func__
Event.args   = (msg,)             # explicit, serializable payload
```

- **Single-process (now):** just call `action(*args, **kwargs)`.
- **Parallel (later):** derive the wire form `(entity.name, "receive", args)` from the
  bound method at the partition boundary.

Practical cost today: prefer `schedule(entity.method, *args)` over
`schedule(lambda: ...)`. Both run fine now; only the former survives partitioning.

## Parallel seams (so "later" is not a rewrite)

- **Named address space** — every entity unique in `entities`; that name is the future
  cross-rank address.
- **Explicit args in the Event IR** — serializable payload, no hidden closure state.
- **Queue behind a Protocol** — swap in a synchronization-window queue.
- **`run_until(t)`** — already the conservative-synchronization / lookahead primitive.
- **`quantum_manager` slot** — reserved on the Timeline, unimplemented.

## Deferred / out of scope for now

- **Event cancellation and rescheduling** — not a rudimentary requirement. When added it
  brings back a `valid` flag + lazy deletion (or `decrease-key` / `delete` on the queue).
  Deliberately left out of the IR for now to keep Event and EventQueue minimal.
- Quantum state manager (formalisms: ket / density matrix / stabilizer).
- The parallel kernel itself (sync windows, remote entity proxies, quantum-manager server).
- Detailed designs for Event, EventQueue, Entity.

## File layout (`src/qetwork/kernel/`)

```
kernel/
  timeline.py      # Timeline — the engine (focus)
  event.py         # Event
  event_queue.py   # EventQueue
  entity.py        # Entity (abstract base)
```
