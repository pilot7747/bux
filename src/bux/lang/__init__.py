"""The .bux architecture language: a small algebra for wiring layers into blocks and models.

    norm  = rmsnorm(eps=1e-5)
    attn  = gqa(num_heads=32, num_kv_heads=8, rope=rope(base=500000))
    mlp   = gated_mlp(intermediate_size=14336, activation=swish)
    block = [norm attn] [norm mlp]          # [f] is a residual branch: x + f(x)
    model = embed block^32 norm head

See docs/language.md for the full reference.
"""

from bux.lang.compiler import Report, check, compile, compile_file
from bux.lang.ir import Architecture
from bux.lang.syntax import BuxError, parse

__all__ = [
    "Architecture",
    "BuxError",
    "Report",
    "check",
    "compile",
    "compile_file",
    "parse",
]
