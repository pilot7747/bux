"""Overlap: XLA's options as the `compiler` setting (bux.train.xla), the time a step's
collectives run alone in a profiler's trace (bux.train.trace), and MFU that counts the
attention a mask keeps (bux.train.flops)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.profiler import ProfileData
from jaxtyping import Int

from bux.lang import BuxError, compile_runs
from bux.parallel import methods
from bux.parallel.context import AttentionArgs
from bux.train import config as tcfg
from bux.train import flops, trace, xla

MODEL = """
d_model = 32
vocab_size = 64
norm = rmsnorm
attn = gqa(num_heads=4, num_kv_heads=2)
ffn = gated_mlp(intermediate_size=64, activation=swish)
block = [norm attn] [norm ffn]
model = embed block^2 norm head
"""


def run_with(*settings: str):
    given = ["data = synthetic", "optimizer = adamw(lr=1e-3)", "seq_len = 16"]
    given += ["batch = 8", "steps = 3"]
    [run] = compile_runs(MODEL + "\n".join([*given, *settings]))
    return run


# --- the compiler setting -------------------------------------------------------------


@pytest.mark.parametrize("preset", [p for p in xla.PRESETS if p != "default"])
def test_a_run_trains_the_same_by_every_preset_and_keeps_its_id(losses, preset):
    split = {"mesh": "mesh(fsdp=4, tensor=2)", "layout": "layout(stream=tensor)"}
    expected = losses(MODEL, **split)
    got = losses(MODEL, compiler=f"compiler(preset={preset})", **split)
    np.testing.assert_allclose(got, expected, rtol=1e-6)
    assert run_with(f"compiler = compiler(preset={preset})").id == run_with().id


def test_options_go_over_the_preset_s_and_true_and_false_are_bools():
    config = tcfg.CompilerConfig(
        preset="maxtext",
        options="--xla_gpu_enable_triton_gemm=true xla_gpu_memory_limit_slop_factor=95",
    )
    options = xla.options(config)
    assert options["xla_gpu_enable_triton_gemm"] is True
    assert options["xla_gpu_enable_latency_hiding_scheduler"] is True
    assert options["xla_gpu_memory_limit_slop_factor"] == "95"
    assert xla.options(tcfg.CompilerConfig()) == {}


def test_an_unknown_preset_names_the_known_ones():
    with pytest.raises(ValueError, match="none of default, o1, lhs"):
        xla.options(tcfg.CompilerConfig(preset="o3"))


def test_an_option_is_a_name_and_a_value():
    with pytest.raises(BuxError, match="an option is --name=value"):
        run_with('compiler = compiler(options="--xla_gpu_enable_triton_gemm")')


def test_a_step_compiled_with_options_compiles_again_for_other_shapes():
    step = xla.jit(lambda x: 2 * x, {"xla_gpu_enable_triton_gemm": False})
    assert isinstance(step, xla.Compiled)
    np.testing.assert_array_equal(step(jnp.ones(3)), 2 * np.ones(3))
    np.testing.assert_array_equal(step(jnp.ones((2, 2))), 2 * np.ones((2, 2)))
    assert len(step.compiled) == 2


# --- exposed communication ------------------------------------------------------------


def plane(gpu: int, streams: dict[str, list[tuple[str, int, int]]]) -> str:
    """The text of an XPlane of GPU `gpu`: a line per stream, of kernels (name, start
    and end in µs), each with its XLA op by its name."""
    names = sorted({k for kernels in streams.values() for k, _, _ in kernels})
    ids = {name: i + 1 for i, name in enumerate(names)}
    lines = [
        f'lines {{ id: {i} name: "{stream}" timestamp_ns: 0 '
        + " ".join(
            f"events {{ metadata_id: {ids[k]} offset_ps: {s * 10**6} "
            f'duration_ps: {(e - s) * 10**6} stats {{ metadata_id: 99 str_value: "{k}" }} }}'
            for k, s, e in kernels
        )
        + " }"
        for i, (stream, kernels) in enumerate(streams.items())
    ]
    metadata = [
        f'event_metadata {{ key: {i} value {{ id: {i} name: "{k}" }} }}'
        for k, i in ids.items()
    ]
    stat = 'stat_metadata { key: 99 value { id: 99 name: "hlo_op" } }'
    return (
        f'planes {{ id: {gpu + 1} name: "/device:GPU:{gpu}" '
        + " ".join([*lines, *metadata, stat])
        + " }"
    )


def test_exposed_comm_is_the_collectives_time_with_no_compute_beside_them(tmp_path):
    # GPU 0, two steps: a gather from 0 to 40 µs beside a matmul until 30 and a copy
    # (neither) after; a reduce-scatter from 50 to 60 beside nothing; and a fusion on
    # another stream from 55 to 58 that hides 3 µs of it. GPU 1: a matmul alone.
    gpu0 = plane(
        0,
        {
            "Stream #13(compute)": [("gemm", 0, 30), ("MemcpyD2D", 30, 45)],
            "Stream #20(nccl)": [
                ("all-gather-start.1", 0, 40),
                ("ncclDevKernel_ReduceScatter", 50, 60),
            ],
            "Stream #14(compute)": [("fusion.2", 55, 58)],
            "XLA Ops": [("all-gather-start.1", 0, 40)],  # not a stream: left out
        },
    )
    gpu1 = plane(1, {"Stream #13(compute)": [("gemm", 0, 30)]})
    file = tmp_path / "host.xplane.pb"
    file.write_bytes(ProfileData.text_proto_to_serialized_xspace(gpu0 + gpu1))
    assert trace.latest(tmp_path) == file
    got = trace.comm_ms(file, steps=2)
    # GPU 0: 50 µs of collectives, 10 + 7 of them alone; GPU 1: none. A step, a GPU.
    assert got == pytest.approx({"comm_ms": 50e-3 / 4, "exposed_comm_ms": 17e-3 / 4})


def test_a_trace_without_gpus_reports_nothing(tmp_path):
    file = tmp_path / "host.xplane.pb"
    cpu = 'planes { id: 1 name: "/host:CPU" }'
    file.write_bytes(ProfileData.text_proto_to_serialized_xspace(cpu))
    assert trace.comm_ms(file, steps=1) == {}


def test_intervals_merge_and_overlap():
    assert trace.merged([(5, 7), (0, 2), (1, 3), (7, 8)]) == [(0, 3), (5, 8)]
    assert trace.overlap([(0, 3), (5, 8)], [(2, 6), (7, 9)]) == 1 + 1 + 1


# --- MFU ------------------------------------------------------------------------------


def one_document(t: int) -> Int[np.ndarray, "1 t"]:
    return np.arange(t)[None]


@pytest.mark.parametrize("window", [None, 5, 16, 40])
def test_the_causal_share_is_the_logits_a_query_sees_of_t_by_t(window):
    t = 16
    seen = np.tril(np.ones((t, t)))
    if window is not None:
        seen = np.triu(seen, -(window - 1))
    within = flops._within(one_document(t))
    assert flops.Mask(window).share(within) == seen.sum() / t**2


def test_a_mask_sees_each_document_s_own_triangle_and_at_most_its_top_k():
    positions = np.array([[0, 1, 2, 0, 1, 2, 3, 0], [0, 1, 2, 3, 4, 5, 6, 7]])
    within = flops._within(positions)
    seen = np.array([[1, 2, 3, 1, 2, 3, 4, 1], [1, 2, 3, 4, 5, 6, 7, 8]])
    assert flops.Mask().share(within) == seen.mean() / 8
    assert flops.Mask(window=3).share(within) == np.minimum(seen, 3).mean() / 8
    assert flops.Mask(top_k=2).share(within) == np.minimum(seen, 2).mean() / 8
    picks = flops.Mask(top_k=2, picks=True).share(within)
    assert picks == np.minimum(seen, 2).mean() / 2


def xla_attention_flops(window, ids=None):
    b, t, h, g, d = 2, 16, 4, 2, 8
    q = jax.ShapeDtypeStruct((b, t, h, d), jnp.float32)
    kv = jax.ShapeDtypeStruct((b, t, g, d), jnp.float32)
    args = AttentionArgs(ids, window, d**-0.5, None, None)
    xla = methods.attention.find("xla")

    def loss(q, k, v):
        return xla(q, k, v, args).sum()

    backward = jax.grad(loss, argnums=(0, 1, 2))
    every_key = 2 * 2 * b * h * t * t * d  # the logits and the weighted values
    return flops.count(loss, q, kv, kv), flops.count(backward, q, kv, kv), every_key


@pytest.mark.parametrize("window", [None, 4])
def test_attention_counts_only_what_its_mask_keeps_forward_and_backward(window):
    forward, backward, every_key = xla_attention_flops(window)
    assert forward == flops.Work(0.0, {flops.Mask(window): every_key})
    assert backward == flops.Work(0.0, {flops.Mask(window): 3 * every_key})
    share = flops.Mask(window).share(flops._within(one_document(16)))
    assert forward.one_document(16) == share * every_key


def test_attention_on_packed_documents_counts_each_document_s_triangle():
    forward, _, every_key = xla_attention_flops(None)
    positions = np.tile(np.arange(4), 4)[None]  # 4 documents of 4 tokens
    assert forward.of(positions) == every_key * (4 * 10) / 16**2
