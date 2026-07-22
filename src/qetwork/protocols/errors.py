"""Protocol-layer failures: a raised ProtocolError means a logic/desync bug in the
protocol machinery (bad message source, stamp mismatch, illegal re-arm), never a
physics outcome. Exceptions, not asserts -- python -O must not strip these checks."""


class ProtocolError(RuntimeError):
    """A protocol invariant was violated."""
