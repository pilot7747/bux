"""Read and write a layer's ports, the taps through which layers pass values.

A port (``ir.Port``) names a tap, a stack in the forward's `taps`, and a mode: `write`
pushes onto it, `read` takes the top value, and `pop` takes it off.
"""

from collections.abc import Sequence
from typing import Any

from bux.architecture.registry import Taps
from bux.lang import ir


def read(taps: Taps, ports: Sequence[ir.Port], name: str) -> Any:
    """Return the value the layer reads (or pops) from its port `name`."""
    port = next(p for p in ports if p.name == name and p.mode != "write")
    stack = taps[port.tap]
    return stack.pop() if port.mode == "pop" else stack[-1]


def write(taps: Taps, ports: Sequence[ir.Port], name: str, value: object) -> None:
    """Push `value` to the tap of port `name`, if the layer writes that port."""
    for port in ports:
        if port.name == name and port.mode == "write":
            taps.setdefault(port.tap, []).append(value)


def reads(ports: Sequence[ir.Port], name: str) -> bool:
    """Return whether the layer reads (or pops) its port `name`."""
    return any(p.name == name and p.mode != "write" for p in ports)


def writes(ports: Sequence[ir.Port], name: str) -> bool:
    """Return whether the layer writes its port `name`."""
    return any(p.name == name and p.mode == "write" for p in ports)
