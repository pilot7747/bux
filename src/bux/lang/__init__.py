"""The .bux architecture language: a small algebra for wiring layers into blocks and models.

    norm  = rmsnorm(eps=1e-5)
    attn  = gqa(num_heads=32, num_kv_heads=8, rope=rope(base=500000))
    mlp   = gated_mlp(intermediate_size=14336, activation=swish)
    block = [norm attn] [norm mlp]          # [f] is a residual branch: x + f(x)
    model = embed block^32 norm head

A file that also says how to train its model is a run, and its alternatives
(``lr = 1e-3 | 3e-3``) make it a sweep of several. See docs/language.md and
docs/training.md for the full reference.
"""

from bux.lang.compiler import Axis, Report, check, compile, compile_file, compile_runs
from bux.lang.ir import Architecture, Run
from bux.lang.syntax import BuxError, parse

__all__ = [
    "Architecture",
    "Axis",
    "BuxError",
    "Report",
    "Run",
    "check",
    "compile",
    "compile_file",
    "compile_runs",
    "parse",
]
