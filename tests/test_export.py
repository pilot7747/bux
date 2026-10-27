"""Exports to transformers compute what bux does (bux.export): the same logits, the
same greedy tokens with and without the cache, and left-padded batches as each row
alone."""

import ast
import json
import typing
from pathlib import Path

import pytest
from flax import nnx
from transformers.modeling_utils import PreTrainedModel

from bux import config as cfg
from bux.architecture import Model
from bux.export import check, codegen, hf
from bux.export.huggingface import save
from bux.export.plan import RESERVED, plan
from bux.lang import compile

HEAD = "d_model = 64\nvocab_size = 256\n"
ATTN = "attn = gqa(num_heads=4, num_kv_heads=2, rope=rope(fraction=0.5), qk_norm=rmsnorm)\n"
FFN = "ffn = gated_mlp(intermediate_size=128, activation=swish)\n"
BLOCK = HEAD + ATTN + FFN + "block = [rmsnorm attn] [rmsnorm ffn]\n"

WIRINGS = {
    "pre-norm": BLOCK + "model = embed block^2 rmsnorm head",
    "parallel": BLOCK
    + "model = embed block [rmsnorm (attn + ffn)] rmsnorm head(tied=true)",
    "taps": BLOCK + "model = embed @x0 block^2 [$x0 scale(init=0.1)] rmsnorm head",
    "arithmetic": BLOCK + "model = embed [rmsnorm (attn - 0.5 * ffn)] rmsnorm head",
    "shared block": BLOCK.replace("block =", "shared block =")
    + "model = embed block^3 rmsnorm head",
    "shared head": BLOCK
    + "shared out = head\nmodel = embed block @h out $h! block out",
    "learned positions": HEAD
    + FFN
    + "model = embed(max_positions=32) [layernorm(bias=true) ffn] "
    "linear(features=64) gelu layernorm head(bias=true, softcap=30)",
    "nested repeats": BLOCK
    + "model = embed ([rmsnorm attn] [rmsnorm ffn]^2)^2 rmsnorm head",
    "conv": BLOCK
    + "model = embed [conv(size=3, bias=true)] block [rmsnorm conv] rmsnorm head",
    "rope scalings": HEAD
    + "a = gqa(num_heads=4, num_kv_heads=1, rope=rope(layout=interleaved, scaling=pi(factor=4)))\n"
    "b = gqa(num_heads=2, num_kv_heads=2, head_dim=24, rope=rope(base=500000, "
    "scaling=llama3(factor=8, original_max_len=16)))\n"
    "model = embed [rmsnorm a] [rmsnorm(zero_centered=true) b] rmsnorm head",
}


@pytest.mark.parametrize("source", WIRINGS.values(), ids=WIRINGS.keys())
def test_export_computes_what_bux_does(source: str) -> None:
    check.compare(compile(source)).check()


def test_layers_are_named_by_their_labels() -> None:
    p = plan(compile(BLOCK + "model = embed block^4 [rmsnorm ffn]^2 rmsnorm head"))
    paths = [c.path for c in p.calls]
    assert paths[:4] == [
        "model.embed",
        "model.layers.0.rmsnorm",
        "model.layers.0.attn",
        "model.layers.0.rmsnorm_1",
    ]
    assert "model.blocks.1.ffn" in paths and paths[-1] == "model.head"


def test_a_shared_module_is_called_from_each_use() -> None:
    p = plan(
        compile(
            BLOCK.replace("block =", "shared block =") + "model = embed block^3 head"
        )
    )
    attn = [c for c in p.calls if c.layer.config.type == "gqa"]
    assert [c.path for c in attn] == ["model.shared.block.attn"] * 3
    assert [c.first for c in attn] == [True, False, False]


def test_a_label_a_module_class_has_names_its_module_otherwise() -> None:
    """Labels that torch's and transformers' module classes have as attributes take
    a suffix, so that the generated classes keep theirs."""
    source = HEAD + FFN.replace("ffn =", "forward =") + ATTN.replace("attn", "training")
    arch = compile(
        source
        + "config = rmsnorm\nmodel = embed [config training] [config forward] head"
    )
    paths = [c.path for c in plan(arch).calls]
    assert paths[1:4] == ["model.config_1", "model.training_1", "model.config_2"]
    check.compare(arch, implementations=("eager",)).check()


def test_the_reserved_names_cover_the_module_classes() -> None:
    public = {n for n in dir(PreTrainedModel) if not n.startswith("_")}
    assert public <= RESERVED, f"reserve {sorted(public - RESERVED)} in bux.export.plan"


def test_an_export_without_weights_removes_an_earlier_one(tmp_path: Path) -> None:
    arch = compile(HEAD + FFN + "model = embed [rmsnorm ffn] head")
    save(arch, Model(arch, rngs=nnx.Rngs(0)), tmp_path)
    save(arch, None, tmp_path)
    assert not (tmp_path / "model.safetensors").exists()


def test_a_config_constant_never_hides_a_layer_class() -> None:
    source = HEAD + FFN.replace("ffn =", "mlp =") + "model = embed [rmsnorm mlp] head"
    code = codegen.modeling(plan(compile(source)), [])
    assert "MLP_1 = {" in code and "MLP(MLP_1, 64" in code


def test_a_layer_not_exported_yet_says_which(monkeypatch: pytest.MonkeyPatch) -> None:
    classes = {k: v for k, v in codegen.layer_classes().items() if k != "deltanet"}
    monkeypatch.setattr(codegen, "layer_classes", lambda: classes)
    source = HEAD + "model = embed [rmsnorm deltanet(num_heads=2, head_dim=16)] head"
    with pytest.raises(NotImplementedError, match="deltanet"):
        codegen.modeling(plan(compile(source)), [])


@pytest.mark.parametrize(
    "path", sorted(Path(hf.__path__[0]).glob("*.py")), ids=lambda p: p.name
)
def test_layer_files_run_where_transformers_does(path: Path) -> None:
    """The copied files parse on Python 3.10 and import nothing of bux."""
    tree = ast.parse(path.read_text(), feature_version=(3, 10))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            assert not (node.module or "").startswith("bux"), node.module
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith("bux") for a in node.names)


# Every setting of every config the export knows: a config that gains one fails
# test_every_setting_is_known_to_the_export, so that no setting goes unexported unseen.
SETTINGS = Path(__file__).with_name("export_settings.json")


def test_every_setting_is_known_to_the_export() -> None:
    configs = (*typing.get_args(typing.get_args(cfg.LayerConfig)[0]),)
    configs += typing.get_args(typing.get_args(cfg.SubConfig)[0])
    known = json.loads(SETTINGS.read_text())
    for c in configs:
        new = set(c.model_fields) - set(known.get(c.__name__, ()))
        gone = set(known.get(c.__name__, ())) - set(c.model_fields)
        assert not new and not gone, (
            f"{c.__name__} gained {sorted(new)} and lost {sorted(gone)}: export the "
            "new settings in bux.export.hf (or refuse them there with unsupported), "
            f"then update {SETTINGS.name}"
        )
