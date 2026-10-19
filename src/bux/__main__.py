"""bux check|render|fmt|json FILE.bux [--entry NAME]"""

import argparse
import sys
from collections import Counter

from bux.lang import Architecture, check
from bux.lang.printer import to_source
from bux.lang.render import render


def summary(arch: Architecture) -> str:
    counts = Counter(layer.config.type for layer in arch.layers())
    head = (
        f"d_model={arch.d_model} vocab_size={arch.vocab_size} layers={counts.total()}"
    )
    if arch.residual is not None:
        head += f" residual={arch.residual.type}(expansion={arch.residual.expansion})"
    return head + "\n" + "  ".join(f"{t}×{n}" for t, n in counts.most_common())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bux", description="Compile and inspect .bux architectures."
    )
    parser.add_argument("command", choices=["check", "render", "fmt", "json"])
    parser.add_argument("file")
    parser.add_argument(
        "--entry", default="model", help="definition to compile (default: model)"
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="render: draw blocks inline instead of separately",
    )
    args = parser.parse_args(argv)
    with open(args.file, encoding="utf-8") as f:
        report = check(f.read(), entry=args.entry, filename=args.file)
    if report.arch is None:
        print("\n\n".join(map(str, report.errors)), file=sys.stderr)
        return 1
    arch = report.arch
    match args.command:
        case "check":
            print(summary(arch))
        case "render":
            print(render(arch, expand=args.expand), end="")
        case "fmt":
            print(to_source(arch), end="")
        case "json":
            print(arch.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
