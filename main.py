#!/usr/bin/env python3
"""Entry script for Contra‑PPO.

Usage examples
--------------
# train
python main.py train

# 5 stochastic evaluation (default 5 episodes)
python main.py evaluate

# 10 stochastic evaluation
python main.py evaluate -e 10

# 1 deterministic evaluation (default 1 episode)
python main.py evaluate --deterministic
"""
from __future__ import annotations

import argparse
import pathlib
import sys

from src.config import TrainingConfig as Cfg
from src.trainer import PPOTrainer
from src.evaluate import evaluate as run_evaluate


def default_ckpt(is_deterministic) -> pathlib.Path:
    if is_deterministic:
        path = pathlib.Path("trained_models") / "contra_deterministic.pth"
    else:
        path = pathlib.Path("trained_models") / "contra_stochastic.pth"
    if not path.exists():
        sys.exit(f"Can not find checkfiles: {path}")
    return path


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Contra‑PPO CLI")
    sub = parser.add_subparsers(dest="mode", required=True)

    # ---- train ----
    sub.add_parser("train", help="start training loop")

    # ---- evaluate ----
    eva = sub.add_parser("evaluate", help="run evaluation episodes")
    eva.add_argument(
        "-e", "--episodes",
        type=int,
        default=5,
        help="number of episodes to run (default: 5)",
    )
    eva.add_argument(
        "--deterministic",
        action="store_true",
        help="deterministic evaluation (skip=0); "
             "omit for stochastic eval (skip=2)",
    )
    return parser


def main() -> None:
    args = build_cli().parse_args()

    if args.mode == "train":
        PPOTrainer(Cfg).train()
        return

    # ---------- evaluate ----------
    if args.deterministic:
        print("Running deterministic evaluation episode = 1")
        run_evaluate(
        ckpt=str(default_ckpt(True)),
        episodes=1,
        skip=0,
    )
    else:
        print(f"Running stochastic evaluation for {args.episodes} episodes")
        run_evaluate(
        ckpt=str(default_ckpt(False)),
        episodes=args.episodes,
        skip=2,
    )


if __name__ == "__main__":
    main()
