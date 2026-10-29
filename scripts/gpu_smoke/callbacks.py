"""cuDNN's GDN and GDN-2 kernels, which run from a ``buffer_callback``, between
collectives on 2 GPUs: a scan of layers, each its kernel on a device's rows, then a
mean over every device's rows (an all-reduce), forward and backward, three steps.
A graph's first execute on a device compiles and loads its kernels, which waits for
the device's peers: inside a step, a peer waiting at a collective this device joins
only after the callback returned hung the step, until ``fused_delta.warm`` ran each
graph on every device first. A step that does not finish in `TIMEOUT` seconds is a
hang; the check runs in a process of its own, which a hang leaves behind.

    uv run --extra cuda python scripts/gpu_smoke/callbacks.py

It needs 2 B200s or B300s and cudnn-frontend, and exits 1 if a step hangs.
"""

import subprocess
import sys

TIMEOUT = 600  # seconds for the warm-ups, three steps and their compiles
LAYERS = 4

STEP = f"""
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from bux import parallel
from bux.architecture.recurrent import fused_delta, linear_attention
from bux.train import config as tcfg

b, t, h, d = 4, 2048, 4, 128
assert fused_delta.runs(d, d, jnp.bfloat16, 1, None), "no cuDNN GDN kernel here"
gdn2 = fused_delta.linear_runs(d, d, jnp.bfloat16)  # GLA by GDN-2's kernel too, if here
print("GDN-2 kernel:", gdn2, flush=True)
with parallel.use(tcfg.MeshConfig(data=2)):
    rows = NamedSharding(parallel.active().mesh, P("data"))
    keys = jax.random.split(jax.random.key(0), 4)
    q, k, v = (jax.random.normal(key, (b, t, h, d), jnp.bfloat16) for key in keys[:3])
    k = (k / jnp.linalg.norm(k.astype(jnp.float32), axis=-1, keepdims=True)).astype(q.dtype)
    g = -jax.nn.softplus(jax.random.normal(keys[3], (b, t, h, 1)))
    beta = jnp.full((b, t, h), 0.5)
    q, k, v, g, beta = (jax.device_put(x, rows) for x in (q, k, v, g, beta))

    def layer(x, _):
        out = linear_attention.delta_rule(x, k, v, g, beta, None)
        if gdn2:
            out = out + linear_attention.decayed(x, k, v, g, None)
        mean = out.astype(jnp.float32).mean()  # every device's rows: an all-reduce
        return (x + 0.1 * mean * out).astype(x.dtype), mean

    def loss(x):
        return jax.lax.scan(layer, x, None, length={LAYERS})[1].sum()

    step = jax.jit(jax.value_and_grad(loss))
    for i in range(3):
        value, grad = jax.block_until_ready(step(q))
        print(f"step {{i}}: {{float(value):.4f}}", flush=True)
"""


def main() -> int:
    try:
        subprocess.run([sys.executable, "-c", STEP], timeout=TIMEOUT, check=True)
    except subprocess.TimeoutExpired:
        print(f"FAIL: a step hung (no end in {TIMEOUT} s)")
        return 1
    except subprocess.CalledProcessError as e:
        print(f"FAIL: exit {e.returncode}")
        return 1
    print("ok: 3 steps on 2 GPUs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
