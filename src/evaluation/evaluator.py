"""
Evaluation script for trained MADDPG model
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm

from src.environment.exploration_env import GridExplorationEnv
from src.agents.maddpg import MADDPGAgent
from src.models.gnn import build_communication_graph


class Evaluator:
    """
    Evaluate trained MADDPG agent
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_path: str,
        device: str = "cpu",
        render: bool = True,
    ):
        self.config = config
        self.device = device

        # Environment
        env_config = config["environment"]
        agent_config = config["agent"]
        self.env = GridExplorationEnv(
            grid_size=env_config["grid_size"],
            num_agents=env_config["num_agents"],
            num_obstacles=env_config["num_obstacles"],
            communication_range=env_config["communication_range"],
            max_steps=env_config["max_steps"],
            view_range=agent_config["view_range"],
            render_mode="human" if render else None,
            rewards_config=config["rewards"],
        )

        # Agent dimensions
        self.num_agents = env_config["num_agents"]
        dummy_obs, _ = self.env.reset()
        self.obs_dim = dummy_obs[self.env.possible_agents[0]].shape[0]
        self.action_dim = 5

        # MADDPG Agent
        model_config = config["model"]
        self.agent = MADDPGAgent(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            num_agents=self.num_agents,
            actor_lr=model_config["actor_lr"],
            critic_lr=model_config["critic_lr"],
            gamma=model_config["gamma"],
            tau=model_config["tau"],
            gnn_config={
                "gnn_hidden_dim": model_config["gnn_hidden_dim"],
                "gnn_output_dim": model_config["gnn_hidden_dim"],
                "num_gnn_layers": model_config["gnn_num_layers"],
                "num_heads": model_config["gnn_heads"],
            },
            actor_hidden_dims=model_config["actor_hidden_dims"],
            critic_hidden_dims=model_config["critic_hidden_dims"],
            device=device,
        )

        # Load trained model
        print(f"Loading model from {model_path}")
        self.agent.load(model_path)
        self.agent.actors[0].eval()  # Set to evaluation mode

        self.communication_range = env_config["communication_range"]

    def evaluate(self, num_episodes: int = 100, render: bool = True) -> Dict[str, Any]:
        """
        Evaluate the agent

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        print(f"\nEvaluating for {num_episodes} episodes...")

        episode_rewards = []
        episode_exploration_rates = []
        episode_steps_list = []
        collision_counts = []

        for episode in tqdm(range(num_episodes)):
            obs_dict, _ = self.env.reset(
                seed=episode + 1000
            )  # Different seeds from training
            obs = np.array([obs_dict[agent] for agent in self.env.agents])

            episode_reward = 0.0
            episode_steps = 0
            collision_count = 0
            done = False

            while not done:
                # Get agent positions
                positions = np.array(
                    [self.env.agent_positions[agent] for agent in self.env.agents]
                )
                adjacency = build_communication_graph(
                    positions,
                    self.communication_range,
                    self.num_agents,
                )

                # Select actions (no exploration)
                actions = self.agent.get_actions(obs, adjacency, epsilon=0.0)
                actions_dict = {
                    agent: actions[i] for i, agent in enumerate(self.env.agents)
                }

                # Step
                next_obs_dict, rewards_dict, terminations, truncations, infos = (
                    self.env.step(actions_dict)
                )

                # Convert to arrays
                next_obs = np.array([next_obs_dict[agent] for agent in self.env.agents])
                rewards = np.array([rewards_dict[agent] for agent in self.env.agents])

                # Count collisions (negative rewards from collision)
                collision_count += sum(1 for r in rewards if r < -0.3)

                # Update
                obs = next_obs
                episode_reward += rewards.mean()
                episode_steps += 1

                # Render
                if render and episode < 5:  # Only render first 5 episodes
                    self.env.render()

                # Check done
                done = any(terminations.values()) or any(truncations.values())

            # Record metrics
            exploration_rate = infos[self.env.agents[0]]["exploration_rate"]
            episode_rewards.append(episode_reward)
            episode_exploration_rates.append(exploration_rate)
            episode_steps_list.append(episode_steps)
            collision_counts.append(collision_count)

        # Calculate statistics
        metrics = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_exploration_rate": np.mean(episode_exploration_rates),
            "std_exploration_rate": np.std(episode_exploration_rates),
            "mean_steps": np.mean(episode_steps_list),
            "mean_collisions": np.mean(collision_counts),
            "max_exploration_rate": np.max(episode_exploration_rates),
            "min_exploration_rate": np.min(episode_exploration_rates),
        }

        # Print results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(
            f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}"
        )
        print(
            f"Mean Exploration Rate: {metrics['mean_exploration_rate']:.2%} ± {metrics['std_exploration_rate']:.2%}"
        )
        print(f"Max Exploration Rate: {metrics['max_exploration_rate']:.2%}")
        print(f"Min Exploration Rate: {metrics['min_exploration_rate']:.2%}")
        print(f"Mean Steps per Episode: {metrics['mean_steps']:.1f}")
        print(f"Mean Collisions per Episode: {metrics['mean_collisions']:.1f}")
        print("=" * 60 + "\n")

        # Save plots
        self._plot_results(
            episode_rewards,
            episode_exploration_rates,
            episode_steps_list,
        )

        self.env.close()

        return metrics

    def _plot_results(
        self,
        rewards: List[float],
        exploration_rates: List[float],
        steps: List[int],
    ):
        """Plot evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Rewards
        axes[0, 0].plot(rewards, alpha=0.6)
        axes[0, 0].plot(
            np.convolve(rewards, np.ones(10) / 10, mode="valid"),
            linewidth=2,
            label="Moving Average (10 episodes)",
        )
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Total Reward")
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Exploration Rate
        axes[0, 1].plot(exploration_rates, alpha=0.6, color="green")
        axes[0, 1].plot(
            np.convolve(exploration_rates, np.ones(10) / 10, mode="valid"),
            linewidth=2,
            label="Moving Average (10 episodes)",
            color="darkgreen",
        )
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Exploration Rate")
        axes[0, 1].set_title("Exploration Coverage")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Histogram of exploration rates
        axes[1, 0].hist(exploration_rates, bins=20, edgecolor="black", alpha=0.7)
        axes[1, 0].axvline(
            np.mean(exploration_rates),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(exploration_rates):.2%}",
        )
        axes[1, 0].set_xlabel("Exploration Rate")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Distribution of Exploration Rates")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Steps per episode
        axes[1, 1].plot(steps, alpha=0.6, color="orange")
        axes[1, 1].axhline(
            np.mean(steps),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(steps):.1f}",
        )
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Steps")
        axes[1, 1].set_title("Steps per Episode")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        save_path = Path("results/plots/evaluation_results.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plots saved to {save_path}")

        # Close plot instead of show (non-blocking)
        plt.close()


def evaluate_model(
    config: Dict[str, Any],
    model_path: str,
    num_episodes: int = 100,
    device: str = "cpu",
    render: bool = True,
) -> Dict[str, Any]:
    """Convenience function to evaluate model"""
    evaluator = Evaluator(config, model_path, device, render)
    return evaluator.evaluate(num_episodes, render)
