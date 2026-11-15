"""krt-bench: reproducible benchmark tool for KinetixRT."""

from __future__ import annotations

import argparse
import importlib
import logging
import warnings

COMMANDS = ["model", "signals", "closed_loop", "realtime", "scheduler_sim", "report"]


def main(argv: list[str] | None = None) -> None:
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(prog="krt-bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        importlib.import_module(f"kinetixrt.bench.{name}").add_parser(sub)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
