"""Check vLLM's online FP8 weights, and its FP8 KV cache, on bux's exports served
by bux's vLLM plugin (`bux_vllm`): the same export, its weights in bfloat16, served
in bfloat16 and with ``quantization="fp8"`` (and ``kv_cache_dtype="fp8"``).

It runs in two steps, as `scripts/vllm_parity.py` does (bux and vLLM do not share an
environment):

    # bux's environment: export each case in bfloat16, its weights perturbed, and
    # emulate FP8 on it
    JAX_PLATFORMS=cpu python scripts/vllm_fp8.py export DIR
    # vLLM's, with the plugin (pip install src/bux_vllm): serve each twice, compare
    python scripts/vllm_fp8.py run DIR [--quantization fp8] [--kv-cache-dtype fp8]
        [--moe-backend triton] [--linear-backend cutlass] [--tp 2] [--eager]
        [--only CASE ...]
    # either: judge the runs at DIR again, without serving them
    python scripts/vllm_fp8.py report DIR [--only CASE ...]

`run` serves each case's export twice, each in a process of its own: in bfloat16,
and under ``--quantization`` (``fp8`` by default: vLLM 0.32 quantizes the bfloat16
weights as it loads them, its ``fp8_per_tensor``), and with ``--kv-cache-dtype``
its KV cache in that dtype too; ``--linear-backend`` picks its FP8 GEMMs' kernels
(vLLM's ``linear_backend``: ``cutlass``, ``triton``, ``deep_gemm``...). Each decodes the prompts of
`scripts/vllm_parity.py` greedily and keeps the log-probabilities of the whole
vocabulary at every position. For each case it prints the quantization methods vLLM
picked, counted by kind of layer (its class and the last part of its name), with
their FP8 GEMM; its attention layers' KV cache dtypes and backends; the routed
experts' backend; and how the FP8 logits compare with
bfloat16's at the prompts' positions and at the decoding steps up to the first
token the two runs pick differently (the steps whose prefixes they share): the
cosine of the logits, centered at each position (as the log-probabilities are),
its mean, its median and its least, and the largest error relative to the largest
bfloat16 logit of its position. It exits with 1 if vLLM's FP8 is further off than
`RATIO` times what FP8's rounding alone costs the model (`emulate`), if nothing was
quantized, if a head was, or if no attention layer of a model with some kept the
KV cache asked for.

What FP8's rounding costs is the model's own: these tiny models of random weights
lose far more to it than a trained one, from 0.008 of cosine (GPT-3's) to 0.04 (the
sliding-window milestone's), and at the scale of a weight (per tensor, per channel
or per block of 128) all the same. So ``export`` emulates vLLM's FP8 on the
export's transformers model in float32 (`emulate`): every linear layer vLLM
quantizes rounded to e4m3 per tensor, its inputs per token, the routed experts'
weights per expert and their inputs per tensor, and with an FP8 KV cache the
attention's queries, keys and values at a scale of 1. It writes the cosines of the
emulated logits with the model's own to ``expected.json``. vLLM may be off by up to
`RATIO` (2) times that, plus `FLOOR`, as `scripts/vllm_parity.py` holds bfloat16
to twice bux's own error: vLLM compares with bfloat16 rather than float32, and the
emulation leaves the experts' hidden units and the norms and RoPE after the keys
unrounded. The prompts' mean cosine counts, as the emulation measures it, and the
decoding steps' median: a case decodes few steps up to its first difference, and
where FP8 flips a pick (an expert, an indexer's key) at one of them, that step's
cosine falls far, as it does where the emulation flips one at a prompt's position.
Weights loaded in the wrong place, a scale set by garbage or bytes read in the
wrong format are many times further off. A case whose transformers model does not
run (one of the layers only the plugin exports) has no emulation, and its accuracy
goes unchecked.

The cases are the timeline's milestones at d_model 256 and a vocabulary of 256, the
first group of four layers of their stacks: GPT-3's (2020), a llama, and 2026's two
(the sliding-window camp's, and the hybrid with KDA in place of its windows).
``--only`` picks some of them, or any case of `scripts/vllm_parity.py` by its name
(its ``llama`` is this one), or ``kv-conv-64``, the conv on keys and values at heads
of 64, to check the other families of layers. On a B300 (SM
10.3), FlashInfer's CUTLASS MoE kernels fail, so ``run`` takes vLLM's Triton kernels
by default.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import vllm_parity

HEAD = "d_model = 256\nvocab_size = 256\n"
#: The milestones' SwiGLU of 8/3 the width, in blocks of 64, and an expert an
#: eighth as wide.
WIDTHS = (
    "swiglu_width = (8 * d_model + 96) // 192 * 64\n"
    "expert_width = (swiglu_width + 256) // 512 * 64\n"
)
#: GPT-3's decoder: layernorms and biases, heads of 64, a GELU MLP, learned
#: positions and a tied head.
GPT3 = (
    "norm = layernorm(eps=1e-5, bias=true)\n"
    "attn = gqa(num_heads=d_model // 64, num_kv_heads=d_model // 64, bias=true)\n"
    "ffn = mlp(intermediate_size=4 * d_model, activation=gelu, bias=true)\n"
    "block = [norm attn] [norm ffn]\n"
    "model = embed(max_positions=128) block^2 norm head(tied=true)"
)
LLAMA = (
    "attn = gqa(num_heads=d_model // 64, num_kv_heads=d_model // 128, rope=rope())\n"
    "ffn = gated_mlp(intermediate_size=swiglu_width, activation=swish)\n"
    "block = [rmsnorm attn] [rmsnorm ffn]\n"
    "model = embed block^2 rmsnorm head"
)
#: 2026's sliding-window camp: three windowed layers with RoPE (of 16 keys here,
#: which the prompts cross) to a global one without positions, QK-norm and a gate
#: on every head, a KV head to 4 query heads; the first layer dense, the others
#: routing to 7 of 256 experts by sigmoid scores beside a shared expert; an MTP
#: module, which vLLM does not build.
FRONTIER = (
    "fr_norm = rmsnorm(eps=1e-5)\n"
    "fr_attn = gqa(num_heads=d_model // 64, num_kv_heads=d_model // 256,"
    " qk_norm=fr_norm, gated=true)\n"
    "fr_local = fr_attn(sliding_window_size=16, rope=rope())\n"
    "fr_global = fr_attn\n"
    "fr_dense = gated_mlp(intermediate_size=swiglu_width, activation=swish)\n"
    "fr_experts = moe(num_experts=256, top_k=7, num_shared_experts=1, router=sigmoid,"
    " routed_scaling_factor=2.5, load_balancing=aux_free, aux_loss_coef=1e-4,"
    " aux_loss_scope=sequence,"
    " expert=gated_mlp(intermediate_size=expert_width, activation=swish))\n"
    "fr_layer(mixer, ffn=fr_experts) = [fr_norm mixer] [fr_norm ffn]\n"
    "shared fr_out = head\n"
)
#: The hybrid: KDA in place of the windowed layers, q/k heads of 128 as wide
#: together as the stream, twice as many value heads, its decay bounded at -5.
HYBRID = (
    "hy_delta = kda(num_heads=d_model // 128, num_v_heads=d_model // 64,"
    " head_dim=128, gate_lower_bound=-5)\n"
)

#: The cases, by name: a model each, as bux source.
CASES = {
    "gpt3": HEAD + GPT3,
    "llama": HEAD + WIDTHS + LLAMA,
    "frontier": HEAD
    + WIDTHS
    + FRONTIER
    + "fr_mtp = $h nextn(norm=fr_norm) fr_layer(fr_local) fr_norm fr_out\n"
    "model = embed fr_layer(fr_local, fr_dense) fr_layer(fr_local)^2"
    " fr_layer(fr_global) fr_norm @h fr_out fr_mtp",
    "hybrid": HEAD
    + WIDTHS
    + FRONTIER
    + HYBRID
    + "hy_mtp = $h nextn(norm=fr_norm) fr_layer(fr_global) fr_norm fr_out\n"
    "model = embed fr_layer(hy_delta, fr_dense) fr_layer(hy_delta)^2"
    " fr_layer(fr_global) fr_norm @h fr_out hy_mtp",
}

#: The conv on keys and values at heads of 64 (FlashInfer's, on a B300) beside the
#: value residual, as the serving lanes' global layers have it.
KV_CONV = (
    "attn = gqa(num_heads=4, num_kv_heads=2, kv_conv_size=4, rope=rope())\n"
    "ffn = gated_mlp(intermediate_size=swiglu_width, activation=swish)\n"
    "model = embed [rmsnorm attn(v=@v1)] [rmsnorm ffn] [rmsnorm attn(v=$v1!)]"
    " [rmsnorm ffn] rmsnorm head"
)

#: Every case ``--only`` takes: these, the parity cases, and ``kv-conv-64``.
POOL = vllm_parity.CASES | CASES | {"kv-conv-64": HEAD + WIDTHS + KV_CONV}
RATIO = 2.0  # how many times FP8's own error (emulated) vLLM's may be
FLOOR = 0.002  # and the cosine it may lose besides, where that error is next to none
TAGS = ("bf16", "fp8")  # the two runs of a case, by the name of their files


def main() -> None:
    """Parse the command line and run its step."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="step", required=True)
    for name in ("export", "run", "run-one", "report"):
        s = sub.add_parser(name)
        s.add_argument("dir", type=Path)
        s.add_argument("--only", nargs="*", default=None)
    for name in ("run", "run-one"):
        s = sub.choices[name]
        s.add_argument("--quantization", default="fp8")
        s.add_argument("--kv-cache-dtype", default=None)
        s.add_argument("--linear-backend", default=None)
        s.add_argument("--tp", type=int, default=1)
        s.add_argument("--eager", action="store_true")
        s.add_argument("--moe-backend", default="triton")
        s.add_argument("--mamba-ssm-cache-dtype", default="auto")
        s.add_argument("--attention-backend", default=None)
    sub.choices["run-one"].add_argument("--tag", choices=TAGS, required=True)
    args = p.parse_args()
    cases = list(CASES) if args.only is None else args.only
    if unknown := [c for c in cases if c not in POOL]:
        p.error(f"no such case: {', '.join(unknown)}")
    match args.step:
        case "export":
            for name in cases:
                vllm_parity.export(name, args.dir / name, "bfloat16", POOL)
                expect(args.dir / name, POOL[name])
        case "run":
            bad = [n for n in cases if not run(args, n)]
            sys.exit(1 if bad else 0)
        case "report":
            bad = [n for n in cases if not report(args.dir / n, n)]
            sys.exit(1 if bad else 0)
        case "run-one":
            run_one(args, args.dir / cases[0])


# --- export, in bux's environment: FP8's rounding, emulated ------------------------

E4M3, E4M3_MAX = "float8_e4m3fn", 448.0  # FP8's format, and its largest value


def expect(where: Path, source: str) -> None:
    """Write ``expected.json`` at the export `where` of the model `source`: the mean
    cosine of its logits with FP8 emulated (`emulate`) against its own, in float32
    at the prompts' positions, without an FP8 KV cache (``fp8``) and with one
    (``fp8+kv``); null for a model of layers only the plugin exports, which
    transformers does not run."""
    from bux.lang import compile

    missing = vllm_parity.placeholders(compile(source)).added
    if missing:
        print(f"no emulation of {where.name}: {', '.join(missing)} only in vLLM")
        (where / "expected.json").write_text("null")
        return
    prompts = json.loads((where / "parity.json").read_text())["prompts"]
    own = _logits(where, prompts, None)
    out = {"fp8": False, "fp8+kv": True}
    out = {k: _cosine(_logits(where, prompts, kv), own) for k, kv in out.items()}
    (where / "expected.json").write_text(json.dumps(out))


def _logits(where: Path, prompts: list, kv: bool | None) -> np.ndarray:
    """The logits, (n, vocab), of the export at `where`'s transformers model on
    `prompts` in float32: FP8 emulated (`emulate`, with an FP8 KV cache if `kv`),
    or not (None)."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        where, trust_remote_code=True, dtype=torch.float32
    ).eval()
    if kv is not None:
        emulate(model, kv)
    with torch.no_grad():
        rows = [model(torch.tensor([p])).logits[0, :-1] for p in prompts]
    return torch.cat(rows).double().numpy()


def _cosine(got: np.ndarray, want: np.ndarray) -> float:
    """The mean cosine of the logits `got` with `want`, (n, vocab), centered."""
    c = Closeness()
    c.add(got, want)
    return c.summary()[0]


def emulate(model: Any, kv: bool) -> None:
    """Round `model` (an export's transformers model) as vLLM's online FP8 does: each
    linear layer vLLM quantizes (`_quantized`) its weight per tensor and its input
    per token, the routed experts' weights per expert and their input per tensor;
    with `kv`, attention's queries, keys and values (MLA's latents) at a scale of 1,
    as an FP8 KV cache keeps them."""
    for name, m in list(model.named_modules()):
        if type(m).__name__ == "Linear" and _quantized(model, name):
            m.weight.data = _rounded(m.weight.data, _scale(m.weight.data))
            m.register_forward_pre_hook(lambda _, a: (_per_token(a[0]),))
        if type(m).__name__ == "Experts":
            for p in m.parameters(recurse=False):
                if p.dim() == 3:  # per expert
                    p.data = _stack([_rounded(w, _scale(w)) for w in p.data])
            m.register_forward_pre_hook(
                lambda _, a: (_rounded(a[0], _scale(a[0])), *a[1:])
            )
        if kv and type(m).__name__ in ("Attention", "LatentAttention"):
            for part in ("q", "k", "v", "kv_a", "kv_a_norm"):
                if getattr(m, part, None) is not None:
                    getattr(m, part).register_forward_hook(
                        lambda *a: _rounded(a[2], 1.0)
                    )


def _quantized(model: Any, name: str) -> bool:
    """Whether vLLM quantizes the linear layer `name` of an export's model: not a
    head's or an embedding's, a router, a shared experts' gate, a sparse MLA's
    ``kv_b``, or one of an engram's, a residual algebra's or an indexer's, which
    the plugin keeps in the model's dtype, but a ``gqa`` indexer's of a head dim of
    a multiple of 16 (`_indexer_quantized`)."""
    kept = ("Head", "TokenEmbedding", "Engram")
    parts = name.split(".")
    owners = [model.get_submodule(".".join(parts[:i])) for i in range(len(parts))]
    if any(
        type(o).__name__ in kept or "residual" in type(o).__module__ for o in owners
    ):
        return False
    if any(type(o).__name__ == "Indexer" for o in owners):
        return _indexer_quantized(owners)
    if parts[-1] == "kv_b" and getattr(owners[-1], "indexer", None) is not None:
        return False
    return parts[-1] not in ("router", "shared_gate")


def _indexer_quantized(owners: list) -> bool:
    """Whether vLLM quantizes a projection of the indexer among `owners`: a
    ``gqa`` layer's (an ``Attention``), its head dim a multiple of 16, which the
    plugin merges into one GEMM (one scale for the three, there)."""
    at = next(i for i, o in enumerate(owners) if type(o).__name__ == "Indexer")
    gqa = at > 0 and type(owners[at - 1]).__name__ == "Attention"
    return gqa and owners[at].head_dim % 16 == 0


def _scale(x: Any) -> Any:
    """An FP8 scale of the whole of `x`: its largest magnitude over `E4M3_MAX`."""
    return (x.float().abs().max() / E4M3_MAX).clamp_min(1e-12)


def _per_token(x: Any) -> Any:
    """`x` rounded to FP8 by a scale per token (its last axis)."""
    scale = (x.float().abs().amax(-1, keepdim=True) / E4M3_MAX).clamp_min(1e-12)
    return _rounded(x, scale)


def _rounded(x: Any, scale: Any) -> Any:
    """`x` rounded to FP8 by `scale`, saturating as vLLM's kernels do, in its
    dtype."""
    import torch

    y = (x.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(getattr(torch, E4M3))
    return (y.float() * scale).to(x.dtype)


def _stack(xs: list) -> Any:
    """`xs` stacked."""
    import torch

    return torch.stack(xs)


# --- run, in vLLM's environment ------------------------------------------------------


def run(args: argparse.Namespace, name: str) -> bool:
    """Serve case `name` in bfloat16 and in FP8, each in a process of its own; print
    how they compare, and return whether they agree."""
    for tag in TAGS:
        for old in (args.dir / name).glob(f"{tag}.*"):  # an earlier run's
            old.unlink()
        cmd = [sys.executable, __file__, "run-one", str(args.dir), "--only", name]
        cmd += ["--tag", tag, "--quantization", args.quantization]
        for flag in ("kv_cache_dtype", "linear_backend"):
            if (value := getattr(args, flag)) is not None:
                cmd += [f"--{flag.replace('_', '-')}", value]
        cmd += ["--tp", str(args.tp), "--moe-backend", args.moe_backend]
        cmd += ["--mamba-ssm-cache-dtype", args.mamba_ssm_cache_dtype]
        if args.attention_backend is not None:
            cmd += ["--attention-backend", args.attention_backend]
        subprocess.run(cmd + ["--eager"] * args.eager, check=False)
    return report(args.dir / name, name)


def run_one(args: argparse.Namespace, where: Path) -> None:
    """Serve the export at `where` in bfloat16 or quantized (by ``--tag``), decode
    the prompts (`vllm_parity.decode`), and keep what serves its layers
    (`bux_vllm.served.survey`, which vLLM's workers import) in ``<tag>.json``, with
    the KV cache dtype asked for."""
    from bux_vllm.served import survey

    engine = vllm_parity.engine(args)
    kv = args.kv_cache_dtype if args.tag == "fp8" else None
    if args.tag == "fp8":
        engine["quantization"] = args.quantization
    if kv is not None:
        engine["kv_cache_dtype"] = kv
    if args.tag == "fp8" and args.linear_backend is not None:
        engine["linear_backend"] = args.linear_backend  # the FP8 GEMMs' kernels
    llm = vllm_parity.serve(where, engine)
    vllm_parity.decode(llm, where, args.tag)
    layers = llm.apply_model(survey)[0] | {"kv_cache_dtype": kv}
    (where / f"{args.tag}.json").write_text(json.dumps(layers))


# --- compare ---------------------------------------------------------------------------


class Closeness:
    """How the FP8 logits of some positions compare with bfloat16's: each one's
    cosine, centered, and the largest error relative to its largest logit."""

    def __init__(self) -> None:
        self.cosines: list[np.ndarray] = []
        self.error = 0.0

    def add(self, got: np.ndarray, want: np.ndarray) -> None:
        """Measure the positions of the log-probabilities `got`, (n, vocab), against
        `want`'s, centered at each: the logits centered."""
        a, b = _centered(got), _centered(want)
        dot = (a * b).sum(-1)
        self.cosines.append(
            dot / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)
        )
        if len(a):
            peak = np.abs(b).max(-1)
            self.error = max(self.error, float((np.abs(a - b).max(-1) / peak).max()))

    def summary(self) -> tuple[float, float, float, int]:
        """The mean, the median and the least cosine, and the positions measured."""
        c = np.concatenate(self.cosines) if self.cosines else np.ones(0)
        if not len(c):
            return 1.0, 1.0, 1.0, 0
        return float(c.mean()), float(np.median(c)), float(c.min()), len(c)

    def line(self, what: str) -> str:
        """The comparison as a line of `what` it measured."""
        mean, median, least, n = self.summary()
        return (
            f"{what} ({n}): cosine {mean:.5f} (median {median:.5f}, least"
            f" {least:.5f}), max rel. error {self.error:.3f}"
        )


def _centered(logprobs: np.ndarray) -> np.ndarray:
    """Log-probabilities, (n, vocab), less their mean at each position: the logits
    less theirs, in float64."""
    x = logprobs.astype(np.float64)
    return x - x.mean(-1, keepdims=True)


def report(where: Path, name: str) -> bool:
    """Print how case `name`'s FP8 run at `where` compares with its bfloat16 run: the
    methods vLLM picked, the experts' backends, the logits' closeness; return
    whether it agrees (`_close`, `_quantizes`, `_caches`)."""
    if not all((where / f"{t}.{e}").exists() for t in TAGS for e in ("npz", "json")):
        print(f"{name:10s} no run")
        return False
    layers = {t: json.loads((where / f"{t}.json").read_text()) for t in TAGS}
    fp8, kv = layers["fp8"], layers["fp8"]["kv_cache_dtype"]
    print(f"{name:10s} {where}")
    for kind, n in sorted(fp8["methods"].items()):
        print(f"  {n:4d} {kind}")
    for kind, n in sorted(fp8["caches"].items()):
        print(f"  {n:4d} {kind}")
    for tag in TAGS:
        for backend, n in layers[tag]["backends"].items():
            print(f"  {tag} MoE: {n} x {backend}")
    prompts, steps = closeness(where)
    # The prompts' mean, as the emulation measures it; the decoding steps' median,
    # which a pick that one of their few steps flips does not move
    close = _close(where, kv, [prompts.summary()[0], steps.summary()[1]])
    ok = close and _quantizes(fp8["methods"]) and _caches(fp8["caches"], kv)
    print(f"  {prompts.line('prompts')}")
    print(f"  {steps.line('decoding')}  {'ok' if ok else 'DIFFERS'}")
    return ok


def _close(where: Path, kv: str | None, cosines: list[float]) -> bool:
    """Whether vLLM's `cosines` are within `RATIO` times FP8's own error, and
    `FLOOR`, of the emulation's (``expected.json``, by `kv`); say which. True,
    unchecked, without an emulation."""
    file = where / "expected.json"
    expected = json.loads(file.read_text()) if file.exists() else None
    if expected is None:
        print("  no emulation: accuracy unchecked")
        return True
    own = expected["fp8" if kv is None else "fp8+kv"]
    least = 1 - RATIO * (1 - own) - FLOOR
    print(f"  FP8 emulated: cosine {own:.5f}, so at least {least:.5f}")
    return min(cosines) >= least


def _quantizes(methods: dict[str, int]) -> bool:
    """Whether vLLM quantized some layers by `methods` (`survey`'s), online, and
    neither an embedding nor a head; say so if not."""
    online = [k for k in methods if "Online" in k.split(": ")[-1]]
    heads = [k for k in online if "Embedding" in k or "LMHead" in k]
    if not online or heads:
        print(f"  quantized: {online or 'nothing'}, heads among them: {heads}")
    return bool(online) and not heads


def _caches(caches: dict[str, int], kv: str | None) -> bool:
    """Whether an attention layer of `caches` (`survey`'s) kept its KV cache in
    `kv`, if asked for one and the model has some; say so if not."""
    if kv is None or not caches:
        return True
    if not any(f": kv {kv} " in k for k in caches):
        print(f"  no attention layer kept a {kv} KV cache")
        return False
    return True


def closeness(where: Path) -> tuple[Closeness, Closeness]:
    """The FP8 run's logits at `where` against bfloat16's: at the prompts' positions,
    and at the decoding steps up to the first token they pick differently."""
    meta = json.loads((where / "parity.json").read_text())
    bf16, fp8 = (np.load(where / f"{tag}.npz") for tag in TAGS)
    prompts, steps = Closeness(), Closeness()
    for i, prompt in enumerate(meta["prompts"]):
        n = len(prompt) - 1  # the prompt's rows, then a row a step
        want, got = bf16[f"logprobs{i}"], fp8[f"logprobs{i}"]
        prompts.add(got[:n], want[:n])
        k = n + _shared(bf16[f"tokens{i}"], fp8[f"tokens{i}"])
        steps.add(got[n:k], want[n:k])
    return prompts, steps


def _shared(a: np.ndarray, b: np.ndarray) -> int:
    """The decoding steps two runs' tokens `a` and `b` share a prefix for: up to
    their first difference, that step included."""
    differ = np.flatnonzero(a != b)
    return int(differ[0]) + 1 if differ.size else len(a)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")  # apply_model's
    main()
