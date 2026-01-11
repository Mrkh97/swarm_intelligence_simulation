"""
Training loop for MADDPG
"""

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any
from torch.utils.tensorboard import SummaryWriter

from src.environment.exploration_env import GridExplorationEnv
from src.agents.maddpg import MADDPGAgent
from src.utils.replay_buffer import ReplayBuffer
from src.models.gnn import build_communication_graph


class Trainer:
    """
    MADDPG Trainer for multi-agent exploration
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cpu",
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
            render_mode="human" if config["visualization"]["render_train"] else None,
            rewards_config=config["rewards"],
        )

        # Agent dimensions
        self.num_agents = env_config["num_agents"]
        dummy_obs, _ = self.env.reset()
        self.obs_dim = dummy_obs[self.env.possible_agents[0]].shape[0]
        self.action_dim = 5  # Up, Down, Left, Right, Stay

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

        # Replay buffer
        training_config = config["training"]
        self.replay_buffer = ReplayBuffer(
            capacity=training_config["buffer_size"],
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
        )

        # Training parameters
        self.num_episodes = training_config["num_episodes"]
        self.batch_size = training_config["batch_size"]
        self.warmup_steps = training_config["warmup_steps"]
        self.update_frequency = training_config["update_frequency"]
        self.save_frequency = training_config["save_frequency"]

        # Exploration
        self.epsilon = training_config["epsilon_start"]
        self.epsilon_end = training_config["epsilon_end"]
        self.epsilon_decay = training_config["epsilon_decay"]

        # Logging
        self.writer = None
        if config["logging"]["tensorboard"]:
            log_dir = Path(config["logging"]["log_dir"])
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))

        self.print_frequency = config["logging"]["print_frequency"]

        # Communication range
        self.communication_range = env_config["communication_range"]

        # Metrics
        self.episode_rewards = []
        self.episode_exploration_rates = []

    def train(self):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Observation dim: {self.obs_dim}, Action dim: {self.action_dim}")
        print(f"Training for {self.num_episodes} episodes\n")

        total_steps = 0

        for episode in range(self.num_episodes):
            obs_dict, _ = self.env.reset(seed=episode)
            episode_reward = 0.0
            episode_steps = 0
            done = False

            # Convert dict observations to array
            obs = np.array([obs_dict[agent] for agent in self.env.agents])

            while not done:
                # Get agent positions for communication graph
                positions = np.array(
                    [self.env.agent_positions[agent] for agent in self.env.agents]
                )
                adjacency = build_communication_graph(
                    positions,
                    self.communication_range,
                    self.num_agents,
                )

                # Select actions
                if total_steps < self.warmup_steps:
                    # Random actions during warmup
                    actions = np.random.randint(
                        0, self.action_dim, size=self.num_agents
                    )
                else:
                    actions = self.agent.get_actions(
                        obs, adjacency, epsilon=self.epsilon
                    )

                # Convert actions to dict for environment
                actions_dict = {
                    agent: actions[i] for i, agent in enumerate(self.env.agents)
                }

                # Step environment
                next_obs_dict, rewards_dict, terminations, truncations, infos = (
                    self.env.step(actions_dict)
                )

                # Convert to arrays
                next_obs = np.array([next_obs_dict[agent] for agent in self.env.agents])
                rewards = np.array([rewards_dict[agent] for agent in self.env.agents])
                dones = np.array(
                    [
                        terminations[agent] or truncations[agent]
                        for agent in self.env.agents
                    ]
                )

                # Next adjacency matrix
                next_positions = np.array(
                    [self.env.agent_positions[agent] for agent in self.env.agents]
                )
                next_adjacency = build_communication_graph(
                    next_positions,
                    self.communication_range,
                    self.num_agents,
                )

                # Store in replay buffer
                self.replay_buffer.add(
                    obs, actions, rewards, next_obs, dones, adjacency, next_adjacency
                )

                # Update
                if (
                    total_steps >= self.warmup_steps
                    and total_steps % self.update_frequency == 0
                    and self.replay_buffer.is_ready(self.batch_size)
                ):
                    batch = self.replay_buffer.sample(self.batch_size)
                    losses = self.agent.update(batch)

                    if self.writer:
                        self.writer.add_scalar(
                            "train/actor_loss", losses["actor_loss"], total_steps
                        )
                        self.writer.add_scalar(
                            "train/critic_loss", losses["critic_loss"], total_steps
                        )

                # Update state
                obs = next_obs
                episode_reward += rewards.mean()
                episode_steps += 1
                total_steps += 1

                # Check if episode is done
                done = any(terminations.values()) or any(truncations.values())

                # Render
                if (
                    self.config["visualization"]["enabled"]
                    and self.config["visualization"]["render_train"]
                ):
                    if episode % self.config["visualization"]["render_frequency"] == 0:
                        self.env.render()

            # Episode finished
            exploration_rate = infos[self.env.agents[0]]["exploration_rate"]
            self.episode_rewards.append(episode_reward)
            self.episode_exploration_rates.append(exploration_rate)

            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            # Logging
            if self.writer:
                self.writer.add_scalar("episode/reward", episode_reward, episode)
                self.writer.add_scalar(
                    "episode/exploration_rate", exploration_rate, episode
                )
                self.writer.add_scalar("episode/epsilon", self.epsilon, episode)
                self.writer.add_scalar("episode/steps", episode_steps, episode)

            # Print progress
            if episode % self.print_frequency == 0:
                avg_reward = np.mean(self.episode_rewards[-self.print_frequency :])
                avg_exploration = np.mean(
                    self.episode_exploration_rates[-self.print_frequency :]
                )
                print(
                    f"Episode {episode}/{self.num_episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Exploration: {avg_exploration:.2%} | "
                    f"Epsilon: {self.epsilon:.3f} | "
                    f"Steps: {episode_steps} | "
                    f"Buffer: {len(self.replay_buffer)}"
                )

            # Save checkpoint
            if episode % self.save_frequency == 0 and episode > 0:
                save_path = Path("trained_models") / f"maddpg_episode_{episode}.pt"
                self.agent.save(str(save_path))
                print(f"Saved checkpoint to {save_path}")

        # Final save
        final_path = Path("trained_models") / "maddpg_final.pt"
        self.agent.save(str(final_path))
        print(f"\nTraining completed! Final model saved to {final_path}")

        if self.writer:
            self.writer.close()

        self.env.close()

    def close(self):
        """Cleanup"""
        if self.writer:
            self.writer.close()
        self.env.close()


def train_maddpg(config: Dict[str, Any], device: str = "cpu"):
    """Convenience function to train MADDPG"""
    trainer = Trainer(config, device)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        trainer.close()
