"""Check vLLM's online FP8 weights, and its FP8 KV cache, on bux's exports served
by bux's vLLM plugin (`bux_vllm`): the same export, its weights in bfloat16, served
in bfloat16 and with ``quantization="fp8"`` (and ``kv_cache_dtype="fp8"``).

It runs in two steps, as `scripts/vllm_parity.py` does (bux and vLLM do not share an
environment):

    # bux's environment: export each case in bfloat16, its weights perturbed
    JAX_PLATFORMS=cpu python scripts/vllm_fp8.py export DIR
    # vLLM's, with the plugin (pip install src/bux_vllm): serve each twice, compare
    python scripts/vllm_fp8.py run DIR [--quantization fp8] [--kv-cache-dtype fp8]
        [--moe-backend triton] [--tp 2] [--eager] [--only CASE ...]

`run` serves each case's export twice, each in a process of its own: in bfloat16,
and under ``--quantization`` (``fp8`` by default: vLLM 0.32 quantizes the bfloat16
weights as it loads them, its ``fp8_per_tensor``), and with ``--kv-cache-dtype``
its KV cache in that dtype too. Each decodes the prompts of
`scripts/vllm_parity.py` greedily and keeps the log-probabilities of the whole
vocabulary at every position. For each case it prints the quantization methods vLLM
picked, counted by kind of layer (its class and the last part of its name), with
their FP8 GEMM; its attention layers' KV cache dtypes and backends; the routed
experts' backend; and how the FP8 logits compare with
bfloat16's at the prompts' positions and at the decoding steps up to the first
token the two runs pick differently (the steps whose prefixes they share): the
cosine of the logits, centered at each position (as the log-probabilities are),
its mean and its least, and the largest error relative to the largest bfloat16
logit of its position. It exits with 1 if a mean cosine is under `MIN_COSINE` (or
`MIN_COSINE_KV`), if nothing was quantized, if a head was, or if no attention layer
of a model with some kept the KV cache asked for.

`MIN_COSINE` is 0.99. FP8 (e4m3) keeps 3 bits of mantissa: it rounds each weight and
each activation by 3 % RMS, so a matmul's output errs by about 4 %, and the
logits by a few percent, whose cosine is above 0.995; the head stays in bfloat16. A
cosine under 0.99 is an RMS error over 14 %, more than the rounding: a weight loaded
in the wrong place, a wrong scale, or a scale set by garbage.

`MIN_COSINE_KV` is 0.98, for FP8 weights and an FP8 KV cache together. The cache
rounds the keys and values the same way, at a scale of 1 (a bfloat16 checkpoint has
none of its own), and so do the queries where the backend quantizes them to match.
The softmax turns the scores' 5 % error into a larger one on the weights it gives
the values, by the scores' spread: about 5 to 10 % more on each attention layer's
output, in quadrature with the weights' error. 0.98 allows 20 % RMS. Bytes read in
the wrong format, or a scale off by a power of two, are far beyond it.

The cases are the timeline's milestones at d_model 256 and a vocabulary of 256, the
first group of four layers of their stacks: GPT-3's (2020), a llama, and 2026's two
(the sliding-window camp's, and the hybrid with KDA in place of its windows).
``--only`` picks some of them, or any case of `scripts/vllm_parity.py` by its name
(its ``llama`` is this one), to check the other families of layers. On a B300 (SM
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

#: Every case ``--only`` takes: these, and the parity cases.
POOL = vllm_parity.CASES | CASES
MIN_COSINE = 0.99  # of the FP8 logits with bfloat16's, on average (see above)
MIN_COSINE_KV = 0.98  # likewise, with an FP8 KV cache too
TAGS = ("bf16", "fp8")  # the two runs of a case, by the name of their files


def main() -> None:
    """Parse the command line and run its step."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="step", required=True)
    for name in ("export", "run", "run-one"):
        s = sub.add_parser(name)
        s.add_argument("dir", type=Path)
        s.add_argument("--only", nargs="*", default=None)
    for name in ("run", "run-one"):
        s = sub.choices[name]
        s.add_argument("--quantization", default="fp8")
        s.add_argument("--kv-cache-dtype", default=None)
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
        case "run":
            bad = [n for n in cases if not run(args, n)]
            sys.exit(1 if bad else 0)
        case "run-one":
            run_one(args, args.dir / cases[0])


# --- run, in vLLM's environment ------------------------------------------------------


def run(args: argparse.Namespace, name: str) -> bool:
    """Serve case `name` in bfloat16 and in FP8, each in a process of its own; print
    how they compare, and return whether they agree."""
    for tag in TAGS:
        for old in (args.dir / name).glob(f"{tag}.*"):  # an earlier run's
            old.unlink()
        cmd = [sys.executable, __file__, "run-one", str(args.dir), "--only", name]
        cmd += ["--tag", tag, "--quantization", args.quantization]
        if args.kv_cache_dtype is not None:
            cmd += ["--kv-cache-dtype", args.kv_cache_dtype]
        cmd += ["--tp", str(args.tp), "--moe-backend", args.moe_backend]
        cmd += ["--mamba-ssm-cache-dtype", args.mamba_ssm_cache_dtype]
        if args.attention_backend is not None:
            cmd += ["--attention-backend", args.attention_backend]
        subprocess.run(cmd + ["--eager"] * args.eager, check=False)
    return report(args.dir / name, name)


def run_one(args: argparse.Namespace, where: Path) -> None:
    """Serve the export at `where` in bfloat16 or quantized (by ``--tag``), decode
    the prompts (`vllm_parity.decode`), and keep the layers' quantization methods
    and KV caches (`quantized`) in ``<tag>.json``, with the KV cache dtype asked
    for."""
    engine = vllm_parity.engine(args)
    kv = args.kv_cache_dtype if args.tag == "fp8" else None
    if args.tag == "fp8":
        engine["quantization"] = args.quantization
    if kv is not None:
        engine["kv_cache_dtype"] = kv
    llm = vllm_parity.serve(where, engine)
    vllm_parity.decode(llm, where, args.tag)
    layers = llm.apply_model(quantized)[0] | {"kv_cache_dtype": kv}
    (where / f"{args.tag}.json").write_text(json.dumps(layers))


def quantized(model: Any) -> dict:
    """Each kind of layer of `model` (its vLLM class and the last part of its name)
    with the quantization method vLLM picked for it and its FP8 GEMM, counted; its
    attention layers' KV cache dtypes and backends, counted; and the routed
    experts' backends, counted. It runs in vLLM's workers (``LLM.apply_model``)."""
    from collections import Counter

    methods: Counter = Counter()
    backends: Counter = Counter()
    caches: Counter = Counter()
    for name, module in model.named_modules():
        if (kv := getattr(module, "kv_cache_dtype", None)) is not None:
            backend = getattr(module, "attn_backend", None)
            backend = backend.get_name() if backend is not None else None
            caches[f"{type(module).__name__}: kv {kv} ({backend})"] += 1
        method = getattr(module, "quant_method", None)
        if method is None:
            continue
        gemm = getattr(method, "fp8_linear", None)
        how = type(method).__name__ + (f" ({type(gemm).__name__})" if gemm else "")
        methods[f"{type(module).__name__} {name.rsplit('.', 1)[-1]}: {how}"] += 1
        backend = getattr(method, "fp8_backend", None)
        backend = backend or getattr(method, "unquantized_backend", None)
        if backend is not None:
            experts = getattr(getattr(method, "experts_cls", None), "__name__", None)
            backends[f"{getattr(backend, 'value', backend)} ({experts})"] += 1
    return {
        "methods": dict(methods),
        "backends": dict(backends),
        "caches": dict(caches),
    }


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

    def summary(self) -> tuple[float, float, int]:
        """The mean and the least cosine, and the positions measured."""
        c = np.concatenate(self.cosines) if self.cosines else np.ones(0)
        return (float(c.mean()), float(c.min()), len(c)) if len(c) else (1.0, 1.0, 0)

    def line(self, what: str) -> str:
        """The comparison as a line of `what` it measured."""
        mean, least, n = self.summary()
        return (
            f"{what} ({n}): cosine {mean:.5f} (least {least:.5f}),"
            f" max rel. error {self.error:.3f}"
        )


def _centered(logprobs: np.ndarray) -> np.ndarray:
    """Log-probabilities, (n, vocab), less their mean at each position: the logits
    less theirs, in float64."""
    x = logprobs.astype(np.float64)
    return x - x.mean(-1, keepdims=True)


def report(where: Path, name: str) -> bool:
    """Print how case `name`'s FP8 run at `where` compares with its bfloat16 run: the
    methods vLLM picked, the experts' backends, the logits' closeness; return
    whether its mean cosines reach `MIN_COSINE`."""
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
    least = MIN_COSINE if kv is None else MIN_COSINE_KV
    close = min(prompts.summary()[0], steps.summary()[0]) >= least
    ok = close and _quantizes(fp8["methods"]) and _caches(fp8["caches"], kv)
    print(f"  {prompts.line('prompts')}")
    print(f"  {steps.line('decoding')}  {'ok' if ok else 'DIFFERS'}")
    return ok


def _quantizes(methods: dict[str, int]) -> bool:
    """Whether vLLM quantized some layers by `methods` (`quantized`'s), online, and
    neither an embedding nor a head; say so if not."""
    online = [k for k in methods if "Online" in k.split(": ")[-1]]
    heads = [k for k in online if "Embedding" in k or "LMHead" in k]
    if not online or heads:
        print(f"  quantized: {online or 'nothing'}, heads among them: {heads}")
    return bool(online) and not heads


def _caches(caches: dict[str, int], kv: str | None) -> bool:
    """Whether an attention layer of `caches` (`quantized`'s) kept its KV cache in
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
