"""
MADDPG (Multi-Agent Deep Deterministic Policy Gradient) Implementation
GNN ile zenginleştirilmiş Actor-Critic networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.models.gnn import CommunicationGNN


class Actor(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128],
        gnn_hidden_dim: int = 64,
        gnn_output_dim: int = 64,
        num_gnn_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # GNN for communication
        self.gnn = CommunicationGNN(
            input_dim=obs_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            num_layers=num_gnn_layers,
            num_heads=num_heads,
        )

        # MLP layers
        layers = []
        input_dim = obs_dim + gnn_output_dim  # Original obs + GNN output

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Output layer (logits for discrete actions)
        self.output_layer = nn.Linear(input_dim, action_dim)

    def forward(
        self,
        obs: torch.Tensor,
        adjacency_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            obs: [batch_size, num_agents, obs_dim] veya [num_agents, obs_dim]
            adjacency_matrix: [batch_size, num_agents, num_agents] veya [num_agents, num_agents]

        Returns:
            action_logits: [batch_size, num_agents, action_dim] veya [num_agents, action_dim]
        """
        # Handle single batch case
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
            adjacency_matrix = adjacency_matrix.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, num_agents, _ = obs.shape

        # GNN processing
        gnn_output = self.gnn(
            obs, adjacency_matrix
        )  # [batch, num_agents, gnn_output_dim]

        # Concatenate original obs with GNN output
        combined = torch.cat(
            [obs, gnn_output], dim=-1
        )  # [batch, num_agents, obs_dim + gnn_output_dim]

        # MLP
        features = self.mlp(combined)  # [batch, num_agents, hidden_dim]

        # Output logits
        action_logits = self.output_layer(features)  # [batch, num_agents, action_dim]

        if squeeze_output:
            action_logits = action_logits.squeeze(0)

        return action_logits

    def get_action(
        self,
        obs: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        epsilon: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Action selection with epsilon-greedy exploration

        Returns:
            actions: [num_agents] - Selected actions
            action_probs: [num_agents, action_dim] - Action probabilities
        """
        with torch.no_grad():
            action_logits = self.forward(
                obs, adjacency_matrix
            )  # [num_agents, action_dim]
            action_probs = F.softmax(action_logits, dim=-1)

            # Epsilon-greedy
            if np.random.rand() < epsilon:
                actions = torch.randint(0, self.action_dim, (action_logits.shape[0],))
            else:
                actions = torch.argmax(action_probs, dim=-1)

            return actions, action_probs


class Critic(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_dims: List[int] = [256, 256],
        gnn_hidden_dim: int = 64,
        gnn_output_dim: int = 64,
        num_gnn_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents

        # GNN for communication
        self.gnn = CommunicationGNN(
            input_dim=obs_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            num_layers=num_gnn_layers,
            num_heads=num_heads,
        )

        # Centralized input: all agents' obs + GNN + all agents' actions
        input_dim = num_agents * (obs_dim + gnn_output_dim + action_dim)

        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Output: single Q-value
        self.output_layer = nn.Linear(input_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        adjacency_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            obs: [batch_size, num_agents, obs_dim]
            actions: [batch_size, num_agents, action_dim] (one-hot encoded)
            adjacency_matrix: [batch_size, num_agents, num_agents]

        Returns:
            q_values: [batch_size, num_agents] - Q-value for each agent
        """
        batch_size, num_agents, _ = obs.shape

        # GNN processing
        gnn_output = self.gnn(
            obs, adjacency_matrix
        )  # [batch, num_agents, gnn_output_dim]

        # Combine obs + GNN output
        combined_obs = torch.cat(
            [obs, gnn_output], dim=-1
        )  # [batch, num_agents, obs_dim + gnn_output_dim]

        # Flatten all agents' info (centralized)
        combined_obs_flat = combined_obs.reshape(
            batch_size, -1
        )  # [batch, num_agents * (obs_dim + gnn_output_dim)]
        actions_flat = actions.reshape(
            batch_size, -1
        )  # [batch, num_agents * action_dim]

        # Concatenate observations and actions
        critic_input = torch.cat([combined_obs_flat, actions_flat], dim=-1)

        # MLP
        features = self.mlp(critic_input)

        # Q-value
        q_values = self.output_layer(features)  # [batch, 1]

        return q_values


class MADDPGAgent:
    """
    MADDPG Agent: Actor-Critic with GNN
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_agents: int,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        gnn_config: Optional[Dict] = None,
        actor_hidden_dims: List[int] = [128, 128],
        critic_hidden_dims: List[int] = [256, 256],
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # GNN configuration
        gnn_config = gnn_config or {
            "gnn_hidden_dim": 64,
            "gnn_output_dim": 64,
            "num_gnn_layers": 2,
            "num_heads": 4,
        }

        # Create actors for each agent
        self.actors = nn.ModuleList(
            [
                Actor(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_dims=actor_hidden_dims,
                    **gnn_config,
                ).to(device)
                for _ in range(num_agents)
            ]
        )

        self.target_actors = nn.ModuleList(
            [
                Actor(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_dims=actor_hidden_dims,
                    **gnn_config,
                ).to(device)
                for _ in range(num_agents)
            ]
        )

        # Create critics for each agent
        self.critics = nn.ModuleList(
            [
                Critic(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    num_agents=num_agents,
                    hidden_dims=critic_hidden_dims,
                    **gnn_config,
                ).to(device)
                for _ in range(num_agents)
            ]
        )

        self.target_critics = nn.ModuleList(
            [
                Critic(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    num_agents=num_agents,
                    hidden_dims=critic_hidden_dims,
                    **gnn_config,
                ).to(device)
                for _ in range(num_agents)
            ]
        )

        # Copy weights to target networks
        for i in range(num_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

        # Optimizers
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=actor_lr) for actor in self.actors
        ]

        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=critic_lr)
            for critic in self.critics
        ]

    def get_actions(
        self,
        obs: np.ndarray,
        adjacency_matrix: np.ndarray,
        epsilon: float = 0.0,
    ) -> np.ndarray:
        """
        Get actions for all agents

        Args:
            obs: [num_agents, obs_dim]
            adjacency_matrix: [num_agents, num_agents]
            epsilon: Exploration rate

        Returns:
            actions: [num_agents] - Integer actions
        """
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        adj_tensor = torch.FloatTensor(adjacency_matrix).to(self.device)

        # Get all actions at once using the first actor's forward method
        # All actors share the same observation and adjacency
        with torch.no_grad():
            actions = []
            for i, actor in enumerate(self.actors):
                # Each actor gets the full observation but outputs action for all agents
                # We take only the i-th agent's action
                action_logits = actor.forward(
                    obs_tensor.unsqueeze(0),  # [1, num_agents, obs_dim]
                    adj_tensor.unsqueeze(0),  # [1, num_agents, num_agents]
                )  # [1, num_agents, action_dim]

                action_probs = torch.softmax(
                    action_logits[0, i], dim=-1
                )  # [action_dim]

                # Epsilon-greedy
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, self.action_dim)
                else:
                    action = torch.argmax(action_probs).item()

                actions.append(action)

        return np.array(actions, dtype=np.int32)

    def update(
        self,
        batch: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        Update all agents using MADDPG

        Args:
            batch: Dictionary with keys:
                - obs: [batch_size, num_agents, obs_dim]
                - actions: [batch_size, num_agents]
                - rewards: [batch_size, num_agents]
                - next_obs: [batch_size, num_agents, obs_dim]
                - dones: [batch_size, num_agents]
                - adjacency: [batch_size, num_agents, num_agents]
                - next_adjacency: [batch_size, num_agents, num_agents]

        Returns:
            losses: Dictionary of losses
        """
        # Convert to tensors
        obs = torch.FloatTensor(batch["obs"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_obs = torch.FloatTensor(batch["next_obs"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)
        adjacency = torch.FloatTensor(batch["adjacency"]).to(self.device)
        next_adjacency = torch.FloatTensor(batch["next_adjacency"]).to(self.device)

        batch_size = obs.shape[0]

        # One-hot encode actions
        actions_onehot = F.one_hot(actions, num_classes=self.action_dim).float()

        actor_losses = []
        critic_losses = []

        # Update each agent
        for agent_idx in range(self.num_agents):
            # === Update Critic ===
            with torch.no_grad():
                # Get next actions from target actors
                next_actions_list = []
                for i, target_actor in enumerate(self.target_actors):
                    next_action_logits = target_actor(next_obs, next_adjacency)
                    next_action_probs = F.softmax(next_action_logits, dim=-1)
                    next_actions_list.append(
                        next_action_probs[:, i, :]
                    )  # [batch, action_dim]

                next_actions = torch.stack(
                    next_actions_list, dim=1
                )  # [batch, num_agents, action_dim]

                # Get target Q-values
                target_q = self.target_critics[agent_idx](
                    next_obs, next_actions, next_adjacency
                )

                # Compute target
                target_value = (
                    rewards[:, agent_idx : agent_idx + 1]
                    + self.gamma * (1 - dones[:, agent_idx : agent_idx + 1]) * target_q
                )

            # Current Q-value
            current_q = self.critics[agent_idx](obs, actions_onehot, adjacency)

            # Critic loss
            critic_loss = F.mse_loss(current_q, target_value)

            # Update critic
            self.critic_optimizers[agent_idx].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent_idx].parameters(), 1.0)
            self.critic_optimizers[agent_idx].step()

            critic_losses.append(critic_loss.item())

            # === Update Actor ===
            # Get current actions from all actors
            current_actions_list = []
            for i, actor in enumerate(self.actors):
                if i == agent_idx:
                    action_logits = actor(obs, adjacency)
                    action_probs = F.softmax(action_logits, dim=-1)
                    current_actions_list.append(action_probs[:, i, :])
                else:
                    # Use actual actions for other agents (MADDPG trick)
                    current_actions_list.append(actions_onehot[:, i, :])

            current_actions = torch.stack(
                current_actions_list, dim=1
            )  # [batch, num_agents, action_dim]

            # Actor loss: negative Q-value
            actor_q = self.critics[agent_idx](obs, current_actions, adjacency)
            actor_loss = -actor_q.mean()

            # Update actor
            self.actor_optimizers[agent_idx].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), 1.0)
            self.actor_optimizers[agent_idx].step()

            actor_losses.append(actor_loss.item())

        # Soft update target networks
        self.soft_update_targets()

        return {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
        }

    def soft_update_targets(self):
        """Soft update target networks"""
        for i in range(self.num_agents):
            # Update target actor
            for target_param, param in zip(
                self.target_actors[i].parameters(),
                self.actors[i].parameters(),
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            # Update target critic
            for target_param, param in zip(
                self.target_critics[i].parameters(),
                self.critics[i].parameters(),
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def save(self, path: str):
        """Save all networks"""
        torch.save(
            {
                "actors": [actor.state_dict() for actor in self.actors],
                "critics": [critic.state_dict() for critic in self.critics],
                "target_actors": [actor.state_dict() for actor in self.target_actors],
                "target_critics": [
                    critic.state_dict() for critic in self.target_critics
                ],
            },
            path,
        )

    def load(self, path: str):
        """Load all networks"""
        checkpoint = torch.load(path, map_location=self.device)

        for i in range(self.num_agents):
            self.actors[i].load_state_dict(checkpoint["actors"][i])
            self.critics[i].load_state_dict(checkpoint["critics"][i])
            self.target_actors[i].load_state_dict(checkpoint["target_actors"][i])
            self.target_critics[i].load_state_dict(checkpoint["target_critics"][i])
