"""
Replay Buffer for MADDPG
Multi-agent experience replay
"""

import numpy as np
from typing import Dict, Tuple
import random


class ReplayBuffer:
    """
    Experience replay buffer for multi-agent learning
    Stores transitions: (obs, actions, rewards, next_obs, dones, adjacency)
    """

    def __init__(
        self,
        capacity: int,
        num_agents: int,
        obs_dim: int,
        action_dim: int = 1,  # Discrete action (integer)
    ):
        self.capacity = capacity
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.position = 0
        self.size = 0

        # Buffers
        self.obs_buffer = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.action_buffer = np.zeros((capacity, num_agents), dtype=np.int32)
        self.reward_buffer = np.zeros((capacity, num_agents), dtype=np.float32)
        self.next_obs_buffer = np.zeros(
            (capacity, num_agents, obs_dim), dtype=np.float32
        )
        self.done_buffer = np.zeros((capacity, num_agents), dtype=np.float32)
        self.adjacency_buffer = np.zeros(
            (capacity, num_agents, num_agents), dtype=np.float32
        )
        self.next_adjacency_buffer = np.zeros(
            (capacity, num_agents, num_agents), dtype=np.float32
        )

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        adjacency: np.ndarray,
        next_adjacency: np.ndarray,
    ):
        """
        Add a transition to the buffer

        Args:
            obs: [num_agents, obs_dim]
            actions: [num_agents]
            rewards: [num_agents]
            next_obs: [num_agents, obs_dim]
            dones: [num_agents]
            adjacency: [num_agents, num_agents]
            next_adjacency: [num_agents, num_agents]
        """
        self.obs_buffer[self.position] = obs
        self.action_buffer[self.position] = actions
        self.reward_buffer[self.position] = rewards
        self.next_obs_buffer[self.position] = next_obs
        self.done_buffer[self.position] = dones
        self.adjacency_buffer[self.position] = adjacency
        self.next_adjacency_buffer[self.position] = next_adjacency

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample a batch of transitions

        Returns:
            batch: Dictionary with sampled transitions
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            "obs": self.obs_buffer[indices],
            "actions": self.action_buffer[indices],
            "rewards": self.reward_buffer[indices],
            "next_obs": self.next_obs_buffer[indices],
            "dones": self.done_buffer[indices],
            "adjacency": self.adjacency_buffer[indices],
            "next_adjacency": self.next_adjacency_buffer[indices],
        }

    def __len__(self):
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples"""
        return self.size >= batch_size


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (optional advanced feature)
    Samples important transitions more frequently
    """

    def __init__(
        self,
        capacity: int,
        num_agents: int,
        obs_dim: int,
        action_dim: int = 1,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
    ):
        super().__init__(capacity, num_agents, obs_dim, action_dim)

        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance sampling weight
        self.beta_increment = beta_increment
        self.max_priority = 1.0

        # Priority buffer
        self.priorities = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        adjacency: np.ndarray,
        next_adjacency: np.ndarray,
    ):
        """Add with maximum priority"""
        super().add(obs, actions, rewards, next_obs, dones, adjacency, next_adjacency)

        # Set max priority for new experience
        self.priorities[self.position - 1] = self.max_priority

    def sample(
        self, batch_size: int
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Sample with priorities

        Returns:
            batch: Sampled transitions
            indices: Sampled indices (for updating priorities)
            weights: Importance sampling weights
        """
        # Calculate sampling probabilities
        if self.size == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[: self.size]

        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()

        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities)

        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize

        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        batch = {
            "obs": self.obs_buffer[indices],
            "actions": self.action_buffer[indices],
            "rewards": self.reward_buffer[indices],
            "next_obs": self.next_obs_buffer[indices],
            "dones": self.done_buffer[indices],
            "adjacency": self.adjacency_buffer[indices],
            "next_adjacency": self.next_adjacency_buffer[indices],
        }

        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled transitions"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


def test_replay_buffer():
    """Test replay buffer"""
    num_agents = 5
    obs_dim = 100
    capacity = 1000

    buffer = ReplayBuffer(capacity, num_agents, obs_dim)

    # Add some transitions
    for i in range(100):
        obs = np.random.randn(num_agents, obs_dim).astype(np.float32)
        actions = np.random.randint(0, 5, size=num_agents).astype(np.int32)
        rewards = np.random.randn(num_agents).astype(np.float32)
        next_obs = np.random.randn(num_agents, obs_dim).astype(np.float32)
        dones = np.random.randint(0, 2, size=num_agents).astype(np.float32)
        adjacency = np.random.randint(0, 2, size=(num_agents, num_agents)).astype(
            np.float32
        )
        next_adjacency = np.random.randint(0, 2, size=(num_agents, num_agents)).astype(
            np.float32
        )

        buffer.add(obs, actions, rewards, next_obs, dones, adjacency, next_adjacency)

    # Sample batch
    batch = buffer.sample(32)

    print(f"Buffer size: {len(buffer)}")
    print(f"Batch obs shape: {batch['obs'].shape}")
    print(f"Batch actions shape: {batch['actions'].shape}")
    print(f"Batch rewards shape: {batch['rewards'].shape}")
    print("Replay buffer test passed!")


if __name__ == "__main__":
    test_replay_buffer()
