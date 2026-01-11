"""
Multi-Agent Swarm Intelligence Simulation
Main entry point for training and evaluation
"""

import argparse
from pathlib import Path

from src.utils.helpers import load_config, set_seed, create_directories, get_device
from src.training.trainer import train_maddpg
from src.evaluation.evaluator import evaluate_model


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Swarm Intelligence with MADDPG + GNN"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "demo"],
        default="train",
        help="Mode: train, eval, or demo",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="trained_models/maddpg_final.pt",
        help="Path to trained model (for eval/demo mode)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cpu, cuda, or mps (auto-detect if not specified)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (overrides config)"
    )
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Override config with command line arguments
    if args.episodes is not None:
        if args.mode == "train":
            config["training"]["num_episodes"] = args.episodes
        else:
            config["evaluation"]["num_episodes"] = args.episodes

    if args.seed is not None:
        config["seed"] = args.seed

    if args.no_render:
        config["visualization"]["render_train"] = False
        config["visualization"]["render_test"] = False

    # Set seed
    set_seed(config["seed"])
    print(f"Random seed set to {config['seed']}")

    # Create directories
    create_directories(config)

    # Get device
    device = args.device if args.device else get_device()
    print(f"Using device: {device}\n")

    # Run mode
    if args.mode == "train":
        print("=" * 60)
        print("TRAINING MODE")
        print("=" * 60)
        train_maddpg(config, device)

    elif args.mode == "eval":
        print("=" * 60)
        print("EVALUATION MODE")
        print("=" * 60)

        if not Path(args.model).exists():
            print(f"Error: Model file not found: {args.model}")
            print("Please train a model first or specify a valid model path.")
            return

        num_episodes = args.episodes or config.get("evaluation", {}).get(
            "num_episodes", 100
        )
        render = not args.no_render

        evaluate_model(config, args.model, num_episodes, device, render)

    elif args.mode == "demo":
        print("=" * 60)
        print("DEMO MODE - Running single episode")
        print("=" * 60)

        if not Path(args.model).exists():
            print(f"Error: Model file not found: {args.model}")
            print("Please train a model first or specify a valid model path.")
            return

        # Run single episode with rendering
        config["visualization"]["render_test"] = True
        evaluate_model(config, args.model, num_episodes=1, device=device, render=True)


if __name__ == "__main__":
    main()
