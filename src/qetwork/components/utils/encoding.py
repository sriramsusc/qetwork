"""Photonic encodings — the degree of freedom a photon's qubit is carried in.

Shared vocabulary so device declarations and device checks cannot drift apart on
a typo. Matter qubits have no encoding: absorption is a transduction, not a
relabelling, so a memory's stored key carries no photonic DOF at all.
"""

POLARIZATION = "polarization"
TIME_BIN     = "time-bin"      # pulsed early/late bins (kept for future pulsed sources)
ENERGY_TIME  = "energy-time"   # CW Franson: continuous emission-time superposition, analyzed vs an MZI delay
PATH         = "path"

ALL = frozenset({POLARIZATION, TIME_BIN, ENERGY_TIME, PATH})



def validate(encoding: str) -> str:
    if encoding not in ALL:
        raise ValueError(f"unknown encoding {encoding!r}; expected one of {sorted(ALL)}")
    return encoding
