"""The command line's help, as ``bux --help`` prints it."""

from bux.__main__ import parser


def pages() -> dict[str, str]:
    return {"cli.md": f"```text\n{parser().format_help()}```\n"}
