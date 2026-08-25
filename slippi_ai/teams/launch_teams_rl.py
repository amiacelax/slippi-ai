"""
Launch sketch for 2v2 Teams RL — delegates to run_lib.
"""

from __future__ import annotations

import argparse
import sys

from slippi_ai.teams.config import TeamsRLConfig
from slippi_ai.teams.run_lib import run, run_dry


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Teams RL launcher")
    p.add_argument("--teacher", default="models/medium-v2")
    p.add_argument("--hours", type=float, default=3.0)
    p.add_argument("--path", default=None, help="Dolphin folder")
    p.add_argument("--iso", default=None, help="Melee ISO")
    p.add_argument("--dry-run", action="store_true", default=False)
    args = p.parse_args(argv)

    config = TeamsRLConfig(hours=args.hours)
    if args.dry_run or not (args.path and args.iso):
        return run_dry(config)
    return run(config, path=args.path, iso=args.iso)


if __name__ == "__main__":
    sys.exit(main())
