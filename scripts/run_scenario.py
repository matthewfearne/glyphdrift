"""Run a single scenario evaluation."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glyphdrift.evaluation import evaluate
from glyphdrift.scenarios import SCENARIOS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GlyphDrift evaluation")
    parser.add_argument("scenario", choices=list(SCENARIOS.keys()))
    parser.add_argument("--version", default="v0")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    evaluate(args.scenario, version=args.version, episodes=args.episodes)


if __name__ == "__main__":
    main()
