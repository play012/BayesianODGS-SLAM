"""Evaluate a Bayesian omnidirectional GS-SLAM checkpoint.

Usage:
    python run_evaluation.py --checkpoint_path /path/to/run
    python run_evaluation.py --checkpoint_path /path/to/run --novel_view_render
"""
import argparse
from pathlib import Path

from src.evaluation.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a BayesianGS-SLAM reconstruction")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Run directory containing config.yaml and submaps/")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Config YAML (defaults to <checkpoint>/config.yaml)")
    parser.add_argument("--novel_view_render", action="store_true",
                        help="Render every dataset frame instead of keyframes")
    parser.add_argument("--save_render", action="store_true",
                        help="Save rendered images to <checkpoint>/rendered_imgs")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    config_path = Path(args.config_path) if args.config_path else (
        checkpoint_path / "config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find config at {config_path}; pass --config_path")

    evaluator = Evaluator(
        checkpoint_path,
        config_path,
        save_render=args.save_render,
        novel_view_render=args.novel_view_render,
    )
    evaluator.run()


if __name__ == "__main__":
    main()
