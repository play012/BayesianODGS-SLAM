"""Run the Bayesian omnidirectional GS-SLAM pipeline.

Usage:
    python run_slam.py configs/Omni/example.yaml
    python run_slam.py configs/Omni/example.yaml --keyframe_count 4
"""
import argparse
from pathlib import Path

from src.entities.gaussian_slam import GaussianSLAM
from src.utils.io_utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="BayesianGS-SLAM with omnidirectional Gaussians")
    parser.add_argument("config", type=str, help="Path to a scene config YAML")
    parser.add_argument(
        "--keyframe_count", type=int, default=1,
        help="Maximum number of keyframes kept per submap larger than one")
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Override data.output_path from the config")
    parser.add_argument(
        "--disable_wandb", action="store_true",
        help="Disable Weights & Biases logging for this run")
    parser.add_argument(
        "--frame_limit", type=int, default=None,
        help="Override frame_limit (useful for quick smoke tests)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    config["keyframe_count"] = args.keyframe_count
    if args.output_path is not None:
        config.setdefault("data", {})["output_path"] = args.output_path
    if args.disable_wandb:
        config["use_wandb"] = False
    if args.frame_limit is not None:
        config["frame_limit"] = args.frame_limit
        config.setdefault("data", {})["frame_limit"] = args.frame_limit

    config.setdefault("data", {})
    if "input_path" not in config["data"]:
        raise ValueError(
            f"Config {args.config} must define data.input_path")

    slam = GaussianSLAM(config)
    slam.run()
    print(f"Done. Results written to {Path(slam.output_path)}")


if __name__ == "__main__":
    main()
