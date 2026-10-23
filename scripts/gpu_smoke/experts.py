"""Expert parallelism on GPUs, `ragged_all_to_all` above all, which XLA:CPU lacks
(WP7 of plans/parallelism.md).

    uv run --extra cuda python scripts/gpu_smoke/experts.py [--steps 20]

On every visible GPU, an even number of them, each method of methods.moe on each
mesh: the MoE layer's output and gradients against `sorted` on one GPU, then a short
run's losses against one GPU's, and its tokens a second. It exits 1 if the layer or
the first loss differs by more than 1e-4, or any later loss by more than 1e-3:
rounding differs between splits, and the top-k picks and the updates after them
amplify it as the run goes on (examples/experiments/parallel/laptop.bux). If NCCL's
ragged all-to-all hangs, as it was reported to on jax 0.11, try
XLA_FLAGS=--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false.
"""

import argparse
import statistics
import sys
import tempfile

import jax
import numpy as np
from flax import nnx

from bux import config as cfg
from bux import parallel
from bux.architecture import MoE, Taps
from bux.lang import compile_runs
from bux.train import config as tcfg
from bux.train.sinks import Scalars, Sink
from bux.train.trainer import train

METHODS = ["sorted", "all_gather", "all_to_all", "ragged_all_to_all"]
ONE = {"data": 1}
ARCH = """
d_model = 512
vocab_size = 4096
norm = rmsnorm
attn = gqa(num_heads=8, num_kv_heads=4, rope=rope())
experts = moe(num_experts=16, top_k=4, num_shared_experts=1, router=sigmoid,
              load_balancing=aux_free, expert_norm=norm,
              expert=gated_mlp(intermediate_size=512, activation=swish))
model = embed ([norm attn] [norm experts])^2 norm head
data = synthetic
optimizer = adamw(lr=1e-3)
seq_len = 256
log = log(every=1)
"""
EXPERT = cfg.MLPConfig(type="gated_mlp", intermediate_size=128, activation="swish")
NORM = cfg.NormalizationConfig(type="rmsnorm")


class Reports(Sink):
    def __init__(self) -> None:
        self.reports: list[Scalars] = []

    def write(self, step: int, scalars: Scalars) -> None:
        self.reports.append(scalars)


def source(mesh: dict[str, int]) -> str:
    return "mesh(" + ", ".join(f"{a}={n}" for a, n in mesh.items()) + ")"


def layer(mesh: dict[str, int], method: str) -> list[np.ndarray]:
    """The output of an MoE layer, and its gradients, by `method` on `mesh`."""
    c = cfg.MoEConfig(num_experts=16, top_k=4, expert=EXPERT, expert_norm=NORM)
    x = jax.random.normal(jax.random.key(1), (2 * jax.device_count(), 64, 128))
    probe = jax.random.normal(jax.random.key(2), x.shape)
    grid, methods = tcfg.MeshConfig.model_validate(mesh), tcfg.MethodsConfig(moe=method)
    with parallel.use(grid, methods=methods):
        e = parallel.build(lambda: MoE(c, 128, nnx.Rngs(0)))
        y = nnx.jit(lambda e, x: e(x, Taps()))(e, x)
        g = nnx.jit(nnx.grad(lambda e, x: (e(x, Taps()) * probe).sum(), argnums=(0, 1)))
        return jax.tree.leaves(jax.device_get((y, g(e, x))))


def run(mesh: dict[str, int], method: str, steps: int) -> tuple[list[float], float]:
    """A short run's losses by `method` on `mesh`, and its median tokens a second."""
    settings = {
        "mesh": source(mesh),
        "methods": f"methods(moe={method})",
        "steps": steps,
        "batch": 8 * jax.device_count(),
    }
    [r] = compile_runs(ARCH + "".join(f"\n{k} = {v}" for k, v in settings.items()))
    sink = Reports()
    with tempfile.TemporaryDirectory() as out:
        train(r, out, echo=lambda _: None, sinks=[sink])
    losses = [x["loss"] for x in sink.reports]
    speed = statistics.median(x["tokens_per_s"] for x in sink.reports[steps // 2 :])
    return losses, speed


def differs(a: list, b: list) -> float:
    """How far apart two lists of arrays or numbers are at worst, relative to the
    largest magnitude in each of `b`'s."""
    return max(
        float(np.abs(np.subtract(x, y)).max() / np.abs(y).max())
        for x, y in zip(a, b, strict=True)
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=20)
    steps = p.parse_args().steps
    n = jax.device_count()
    if jax.default_backend() == "cpu" or n < 2 or n % 2:
        print(f"needs an even number of GPUs, 2 or more: found {jax.devices()}")
        return 1
    jax.config.update("jax_default_matmul_precision", "highest")
    meshes = [{"expert": n}] + [{"fsdp": 2, "expert": n // 2}] * (n > 2)
    ref, (one, speed) = layer(ONE, "sorted"), run(ONE, "sorted", steps)
    print(f"{source(ONE):28} {'sorted':18} {speed:,.0f} tok/s")
    failed = False
    for mesh in meshes:
        for method in METHODS:
            out = differs(layer(mesh, method), ref)
            losses, speed = run(mesh, method, steps)
            first, loss = differs(losses[:1], one[:1]), differs(losses, one)
            # bfloat16, the default, rounds a run's later losses apart by up to 2e-3.
            failed |= out > 1e-4 or first > 1e-4 or loss > 3e-3
            print(
                f"{source(mesh):28} {method:18} {speed:,.0f} tok/s; off by {out:.1e} "
                f"in the layer, {first:.1e} in the first loss, {loss:.1e} in all",
                flush=True,
            )
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
