"""Where every layer of an architecture goes in an exported model: the module tree
that every exporter shares, so that one set of weights loads into each.

It imports bux.lang, bux.config and numpy alone, no JAX and no torch, so that a
runtime's plugin can place the layers just as `bux.export` named their weights.

The model is a tree of scopes, each a module of its own (a generated class):

- the model's, at ``model.``;
- each repeat's body, a list of them at ``<scope>.<name>.<pass>.``;
- each shared definition's, once, at ``model.shared.<key>``, which every use calls.

A scope names its layers and repeats by their labels (``attn``, ``norm_1``), and the
model's longest repeat ``layers``, as transformers' models name theirs; under a
residual algebra, each residual branch outside any other is a module too, named by
the algebra (``mhc``, ``mhc_1``), which holds the algebra's parameters. `calls`
unrolls the graph into the calls of the layers in the order they run, each with its
module's path, which is also the prefix of its weights.
"""

import keyword
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np

from bux import config as cfg
from bux.config import EngramConfig
from bux.lang import ir

#: A node's place in its scope: the child indices from the scope's root to it.
Pos = tuple[int, ...]

#: Names no module of an export takes: those of what the generated classes and
#: bux_vllm's blocks hold, ``shared`` among them, and the attributes their classes have
#: in torch (``nn.Module``) and transformers (``PreTrainedModel``), which a module of
#: the name would shadow, or be shadowed by. A label among them names its module as
#: a taken name, ``forward_1``. The export writes them in its config
#: (``bux_reserved``), from which bux_vllm's plan takes them, so that the two name
#: the modules alike.
RESERVED = frozenset(
    """
    embed_path head_path layers_by_path plan scope shared vllm_config
    T_destination add_module apply bfloat16 buffers call_super_init children compile
    cpu cuda double dump_patches eval extra_repr float forward get_buffer
    get_extra_state get_parameter get_submodule half ipu load_state_dict modules
    mtia named_buffers named_children named_modules named_parameters parameters
    register_backward_hook register_buffer register_forward_hook
    register_forward_pre_hook register_full_backward_hook
    register_full_backward_pre_hook register_load_state_dict_post_hook
    register_load_state_dict_pre_hook register_module register_parameter
    register_state_dict_post_hook register_state_dict_pre_hook requires_grad_
    set_extra_state set_submodule share_memory smart_apply state_dict to to_empty train
    training type xpu zero_grad
    active_adapters add_adapter add_model_tags all_tied_weights_keys
    barrier_after_gathered_checkpoint_save base_model base_model_prefix can_generate
    can_record_outputs config config_class delete_adapter dequantize device
    disable_adapters disable_input_require_grads dtype dummy_inputs enable_adapters
    enable_input_require_grads enable_peft_hotswap ep_plan from_pretrained fsdp_plan
    fsdp_size gather_sharded_state_dict_for_save generation_config
    generation_config_class get_adapter_state_dict get_compiled_call
    get_correct_attn_implementation get_correct_experts_implementation get_decoder
    get_encoder get_expanded_tied_weights_keys get_experts_implementation
    get_init_context get_input_embeddings get_memory_footprint get_output_embeddings
    get_parameter_or_buffer get_position_embeddings gradient_checkpointing_disable
    gradient_checkpointing_enable init_parallel_plans init_weights
    initialize_weights input_modalities is_backend_compatible is_custom_code
    is_distributed_loading_by_transformers is_gradient_checkpointing is_remote_code
    kernel_config load_adapter loss_function loss_type main_input_name
    mark_tied_weights_as_initialized maybe_distribute_model model_tags name_or_path
    named_non_persistent_buffers num_parameters post_init pp_plan
    prepare_distribute_model push_to_hub register_for_auto_class
    resize_position_embeddings resize_token_embeddings retrieve_modules_from_names
    save_distributed_checkpoint save_pretrained set_adapter set_attn_implementation
    set_decoder set_encoder set_experts_implementation set_input_embeddings
    set_output_embeddings set_use_kernels should_save_on_this_rank
    supports_context_parallel supports_gradient_checkpointing supports_pp_plan
    supports_tp_plan tie_weights tp_plan tp_size use_kernels
    warn_if_padding_and_no_attention_mask warnings_issued
    """.split()  # noqa: SIM905 (as a list, a name a line)
)

#: What a layer type's exporter adds to its layers' `Site` extras, from the plan and
#: the call: ``SITE_EXTRAS["engram"] = lambda plan, call: {"table_sizes": ...}``.
SITE_EXTRAS: dict[str, Callable[["Plan", "LayerCall"], dict]] = {}


@dataclass
class Scope:
    """The modules of one generated class: the model's, a repeat's body, or a shared
    definition's.

    Attributes:
        root: The node the scope computes.
        names: The attribute of each layer, repeat and residual algebra's branch in
            it, by its `Pos`; a shared node in it is a call of ``model.shared``, not
            named here.
        blocks: The scope of each repeat's body, by the repeat's `Pos`; none for a
            repeat of a shared body, which calls it each pass.
        depth: How many residual branches are around its root: for a shared
            definition, around its first use, where bux builds it.
    """

    root: ir.Node
    names: dict[Pos, str] = field(default_factory=dict)
    blocks: dict[Pos, "Scope"] = field(default_factory=dict)
    depth: int = 0

    def name(self, hint: str) -> str:
        """Return a fresh attribute name from `hint`: ``norm``, then ``norm_1``."""
        hint = hint if hint.isidentifier() and not keyword.iskeyword(hint) else "layer"
        taken = set(self.names.values()) | RESERVED
        name, i = hint, 1
        while name in taken:
            name, i = f"{hint}_{i}", i + 1
        return name


@dataclass(frozen=True)
class LayerCall:
    """One call of a layer, as the model runs it.

    Attributes:
        path: The module's path, ``model.layers.3.attn``: the prefix of its weights.
        layer: The layer's node.
        trail: The child indices from the graph's root to the layer, a repeat's pass
            counted as its child: the way to its module in bux's model.
        depth: How many residual branches are around it.
        first: Whether this is its module's first call; a shared module is called
            again, by its later uses.
    """

    path: str
    layer: ir.Layer
    trail: tuple[int, ...]
    depth: int
    first: bool


@dataclass(frozen=True)
class Branch:
    """A residual branch the residual algebra makes, whose module holds the
    algebra's parameters around the branch's body.

    Attributes:
        path: The module's path, ``model.layers.3.mhc``.
        trail: The child indices from the graph's root to the ``ir.Residual``.
        first: Whether this is its module's first call.
    """

    path: str
    trail: tuple[int, ...]
    first: bool


@dataclass
class Plan:
    """An architecture's module tree, and the calls of its layers.

    Attributes:
        arch: The architecture.
        top: The model's scope.
        shared: The scope of each shared definition, by its key.
        calls: Every layer call, in the order they run.
        branches: Every residual algebra's branch, in the order they run.
    """

    arch: ir.Architecture
    top: Scope
    shared: dict[str, Scope]
    calls: list[LayerCall]
    branches: list[Branch] = field(default_factory=list)

    def stop(self) -> Pos | None:
        """Return the item that runs the first head, by its place in the sequences
        around it from the graph's root (none of them shared), if anything runs
        after it: a model asked for the first head's logits alone, as it generates,
        stops after it. None when there is nothing to skip."""
        head = next((c for c in self.calls if c.layer.config.type == "head"), None)
        node, at, after = self.arch.graph, (), False
        for i in () if head is None else head.trail:
            if not isinstance(node, ir.Seq) or node.share is not None:
                break
            at, after = (*at, i), after or i < len(node.items) - 1
            node = node.items[i]
        return at if after else None

    def generation(self) -> list[LayerCall]:
        """Return the calls a model runs for the first head's logits alone: those
        up to `stop`, which keep state in the cache as it generates."""
        stop = self.stop()
        return [c for c in self.calls if stop is None or c.trail[: len(stop)] <= stop]

    def sites(self) -> dict[str, dict]:
        """Return each module's site, by its path: its depth, its ports, and its
        type's `SITE_EXTRAS`."""
        out = {}
        for call in (c for c in self.calls if c.first):
            hook = SITE_EXTRAS.get(call.layer.config.type)
            out[call.path] = {
                "depth": call.depth,
                "ports": [[p.name, p.tap, p.mode] for p in call.layer.ports],
                "extra": hook(self, call) if hook else {},
            }
        return out


def plan(arch: ir.Architecture) -> Plan:
    """Return the module tree of `arch` and the calls of its layers."""
    algebra = None if arch.residual is None else arch.residual.type
    depths = _share_depths(arch.graph, 0, {})
    shared: dict[str, Scope] = {}
    for node in ir.walk(arch.graph):
        if node.share is not None and node.share not in shared:
            depth = depths[node.share]
            shared[node.share] = scope(node, own=True, algebra=algebra, depth=depth)
    own = arch.graph.share is not None
    top = scope(arch.graph, own=own, top=True, algebra=algebra)
    out = Plan(arch, top, shared, [])
    for item in _calls(out, arch.graph, (), top, (), "model.", 0, set()):
        if isinstance(item, LayerCall):
            out.calls.append(item)
        else:
            out.branches.append(item)
    return out


def _share_depths(node: ir.Node, depth: int, out: dict[str, int]) -> dict[str, int]:
    """Return how many residual branches are around each shared definition's first
    use, where bux builds its module."""
    if node.share is not None:
        if node.share in out:
            return out
        out[node.share] = depth
    inner = depth + isinstance(node, ir.Residual)
    for child in ir.children(node):
        _share_depths(child, inner, out)
    return out


def scope(
    root: ir.Node,
    *,
    own: bool,
    top: bool = False,
    algebra: str | None = None,
    depth: int = 0,
) -> Scope:
    """Return the scope of the module computing `root`: `own` says whether `root` is
    the shared node the scope is for, rather than a call of it, `algebra` names the
    residual algebra if any, and `depth` counts the residual branches around it. The
    model's scope (`top`) names its repeat of the most passes ``layers``."""
    s = Scope(root, depth=depth)
    hints: dict[Pos, str] = {}

    def visit(node: ir.Node, pos: Pos, depth: int) -> None:
        if node.share is not None and not (own and pos == ()):
            return
        match node:
            case ir.Layer():
                hints[pos] = node.label or node.config.type
            case ir.Repeat(body=body):
                if body.share is None:
                    hints[pos] = f"{body.label}s" if body.label else "blocks"
                    s.blocks[pos] = scope(body, own=False, algebra=algebra, depth=depth)
            case _:
                if isinstance(node, ir.Residual) and algebra and depth == 0:
                    hints[pos] = algebra
                inner = depth + isinstance(node, ir.Residual)
                for i, child in enumerate(ir.children(node)):
                    visit(child, (*pos, i), inner)

    visit(root, (), depth)
    if top and s.blocks and "layers" not in hints.values():
        hints[max(s.blocks, key=lambda p: _passes(root, p))] = "layers"
    for pos, hint in hints.items():
        s.names[pos] = s.name(hint)
    return s


def _passes(root: ir.Node, pos: Pos) -> int:
    """Return how many passes the repeat at `pos` under `root` makes."""
    for i in pos:
        root = ir.children(root)[i]
    assert isinstance(root, ir.Repeat)
    return root.times


def _calls(
    p: Plan,
    node: ir.Node,
    pos: Pos,
    s: Scope,
    trail: tuple[int, ...],
    prefix: str,
    depth: int,
    seen: set[str],
    first: bool = True,
) -> Iterator[LayerCall | Branch]:
    """Yield the calls of the layers and the algebra's branches under `node`, at
    `pos` in scope `s` whose modules' paths start with `prefix`."""
    key = node.share
    if key is not None and not (s is p.shared.get(key) and pos == ()):
        again = key in seen
        seen.add(key)
        path = f"model.shared.{key}"
        inner = path if isinstance(node, ir.Layer) else f"{path}."
        yield from _calls(
            p, node, (), p.shared[key], trail, inner, depth, seen, first and not again
        )
        return
    match node:
        case ir.Layer():
            path = prefix if pos == () and key is not None else prefix + s.names[pos]
            yield LayerCall(path.rstrip("."), node, trail, depth, first)
        case ir.Repeat(body=body, times=times):
            for i in range(times):
                if body.share is not None:
                    yield from _calls(
                        p, body, (*pos, 0), s, (*trail, i), prefix, depth, seen, first
                    )
                else:
                    inner = f"{prefix}{s.names[pos]}.{i}."
                    yield from _calls(
                        p,
                        body,
                        (),
                        s.blocks[pos],
                        (*trail, i),
                        inner,
                        depth,
                        seen,
                        first,
                    )
        case _:
            if isinstance(node, ir.Residual) and pos in s.names:
                yield Branch(prefix + s.names[pos], trail, first)
            inside = depth + isinstance(node, ir.Residual)
            for i, child in enumerate(ir.children(node)):
                yield from _calls(
                    p, child, (*pos, i), s, (*trail, i), prefix, inside, seen, first
                )


def _next_primes(start: int, count: int, taken: set[int]) -> list[int]:
    """Return the `count` least primes >= start not taken yet, and take them, as
    `bux.architecture.layers.engram` does."""
    found, n = [], max(start, 2)
    while len(found) < count:
        if n not in taken and all(n % p for p in range(2, math.isqrt(n) + 1)):
            found.append(n)
            taken.add(n)
        n += 1
    return found


def engram_hashes(p: Plan, call: LayerCall) -> dict:
    """Return what an engram layer hashes by, as bux builds it: the prime size of each
    n-gram size's and head's table, the least that no engram built before it took,
    and the odd multipliers of its n tokens, drawn from a seed of its index."""
    taken: set[int] = set()
    engrams = [c for c in p.calls if c.first and c.layer.config.type == "engram"]
    for nth, c in enumerate(engrams):
        config = c.layer.config
        assert isinstance(config, EngramConfig)
        n, heads = config.max_ngram_size, config.num_heads
        primes = _next_primes(config.table_size, (n - 1) * heads, taken)
        if c is call:
            odd = np.random.default_rng(10007 * (nth + 1)).integers(0, 2**31, n)
            rows = [primes[i * heads : (i + 1) * heads] for i in range(n - 1)]
            return {"primes": rows, "multipliers": (odd * 2 + 1).tolist()}
    raise AssertionError(f"{call.path} is no engram's first call")


SITE_EXTRAS["engram"] = engram_hashes


def mixer_depth(p: Plan, call: LayerCall) -> dict:
    """Return the index of `call`'s layer among the token mixers the model runs, from
    0, and how many token mixers it runs: what
    ``bux.architecture.recurrent.linear_attention.depth`` builds a layer by."""
    mixers = [
        c.path for c in p.calls if isinstance(c.layer.config, cfg.AttentionConfig)
    ]
    return {"layer": mixers.index(call.path), "layers": max(len(mixers), 1)}


SITE_EXTRAS["lightning"] = mixer_depth  # its decay rates weaken with depth
