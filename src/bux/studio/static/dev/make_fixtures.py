"""Write the diagram harness's fixtures: the studio's ``Model`` of example files.

Each fixture is the ``model`` of the view the server would send for the file
(`bux.studio.view.view`), so the harness draws what the page draws. Run it from the
repository root::

    uv run python src/bux/studio/static/dev/make_fixtures.py
"""

import json
from pathlib import Path

from bux.studio.view import view

HERE = Path(__file__).parent / "fixtures"
EXAMPLES = Path("examples/architectures")
FIXTURES = {
    "llama3_8b": "llama3_8b.bux",
    "qwen3_next": "qwen3_next.bux",
    "hymba": "hymba.bux",
    "mhc": "mhc.bux",
    "deepseek_v3": "deepseek_v3.bux",
    "gpt2": "gpt2.bux",
    "ngpt": "ngpt.bux",
    "modded_nanogpt": "modded_nanogpt.bux",
    "primitives": "primitives.bux",
    "resformer": "resformer.bux",
    "zamba2": "zamba2.bux",
    "qwen3_8_flash_next": "qwen/qwen3.8_flash_next.bux",
    "nemotron_super": "nemotron/nemotron_3_super_120b_a12b.bux",
}
# Shapes no example has: a weighted skip, a scaled branch less a tap, a negation.
ODD = """
d_model    = 64
vocab_size = 100
norm = rmsnorm
attn = gqa(num_heads=4, num_kv_heads=4)
ffn  = mlp(intermediate_size=256, activation=gelu)
deep = (_ * 2.0 + attn) norm
gate = [norm (ffn * 0.5 - $x0)]
model = embed @x0 deep gate [-attn] (norm ffn)^2 norm head
"""


def write(name: str, source: str, path: str) -> None:
    model = view(source, {}, path=path)["model"]
    (HERE / f"model-{name}.json").write_text(json.dumps(model, indent=1) + "\n")
    print(f"model-{name}.json")


def main() -> None:
    HERE.mkdir(exist_ok=True)
    for name, file in FIXTURES.items():
        path = EXAMPLES / file
        write(name, path.read_text(), str(path))
    write("odd", ODD, "odd.bux")


if __name__ == "__main__":
    main()
