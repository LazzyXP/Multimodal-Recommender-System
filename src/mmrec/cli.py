"""Command-line utilities distributed with mmrec."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from mmrec.storage import convert_to_parquet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mmrec")
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert", help="convert CSV/TXT/JSONL data to Parquet")
    convert.add_argument("source")
    convert.add_argument("output", nargs="?")
    convert.add_argument("--delimiter")
    convert.add_argument("--compression", default="zstd")
    convert.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "convert":
        output = convert_to_parquet(
            args.source,
            args.output,
            delimiter=args.delimiter,
            compression=args.compression,
            overwrite=args.overwrite,
        )
        print(output)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
