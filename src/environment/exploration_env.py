"""
Multi-Agent Grid Exploration Environment
5 kara robotu için 2D keşif simülasyonu
"""

import functools
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
import pygame
from typing import Dict, Tuple, Optional, List


class GridExplorationEnv(ParallelEnv):
    """
    5 robotun 2D grid ortamında keşif yapması için environment.

    - Agents: 5 kara robotu
    - Observation: Local grid view + pozisyon + komşu robot bilgileri
    - Action: 0=Up, 1=Down, 2=Left, 3=Right, 4=Stay
    - Reward: Yeni keşif + kollektif bonus - çarpışma cezası
    """

    metadata = {
        "name": "grid_exploration_v0",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        grid_size: int = 25,
        num_agents: int = 5,
        num_obstacles: int = 15,
        communication_range: float = 5.0,
        max_steps: int = 200,
        view_range: int = 3,
        render_mode: Optional[str] = None,
        rewards_config: Optional[Dict] = None,
    ):
        super().__init__()

        self.grid_size = grid_size
        self._num_agents = num_agents  # Private attribute to avoid property conflict
        self.num_obstacles = num_obstacles
        self.communication_range = communication_range
        self.max_steps = max_steps
        self.view_range = view_range
        self.render_mode = render_mode

        # Rewards configuration
        self.rewards_config = rewards_config or {
            "exploration": 1.0,
            "collision": -0.5,
            "step": -0.01,
            "collective_bonus": 2.0,
            "redundant_exploration": -0.1,
        }

        # Agent isimleri
        self.possible_agents = [f"robot_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]

        # Action ve observation spaces
        self._action_spaces = {
            agent: spaces.Discrete(5) for agent in self.possible_agents
        }

        # Observation: [local_view, position, normalized_neighbors]
        # Local view: (2*view_range+1) x (2*view_range+1) x 3 channels
        # Channels: [obstacles, explored, other_robots]
        view_size = 2 * view_range + 1
        obs_size = (view_size * view_size * 3) + 2 + (num_agents - 1) * 3

        self._observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # State variables
        self.grid = None  # Obstacle grid
        self.explored_map = None  # Keşfedilmiş alanlar
        self.agent_positions = {}
        self.step_count = 0

        # Pygame rendering
        self.window = None
        self.clock = None
        self.cell_size = 30

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        """Environment'ı reset et"""
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.step_count = 0

        # Grid oluştur (0: boş, 1: engel)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Random engeller yerleştir
        obstacle_positions = set()
        while len(obstacle_positions) < self.num_obstacles:
            x, y = np.random.randint(0, self.grid_size, 2)
            obstacle_positions.add((x, y))

        for x, y in obstacle_positions:
            self.grid[x, y] = 1

        # Explored map (başlangıçta hiçbir yer keşfedilmemiş)
        self.explored_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Agent pozisyonlarını initialize et (engellerden uzak)
        self.agent_positions = {}
        occupied_positions = obstacle_positions.copy()

        for agent in self.agents:
            while True:
                x, y = np.random.randint(0, self.grid_size, 2)
                if (x, y) not in occupied_positions:
                    self.agent_positions[agent] = np.array([x, y])
                    occupied_positions.add((x, y))
                    # Başlangıç pozisyonunu keşfedilmiş say
                    self.explored_map[x, y] = 1
                    break

        # Observations oluştur
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions: Dict[str, int]):
        """Bir adım simülasyon"""
        self.step_count += 1

        # Her agent için hareket et
        new_positions = {}
        collisions = {agent: False for agent in self.agents}
        newly_explored = {agent: False for agent in self.agents}

        for agent in self.agents:
            action = actions[agent]
            current_pos = self.agent_positions[agent].copy()
            new_pos = self._apply_action(current_pos, action)

            # Grid sınırları kontrolü
            new_pos = np.clip(new_pos, 0, self.grid_size - 1)

            # Engel kontrolü
            if self.grid[new_pos[0], new_pos[1]] == 1:
                collisions[agent] = True
                new_positions[agent] = current_pos  # Hareket etme
            else:
                new_positions[agent] = new_pos

            # Keşif kontrolü
            if self.explored_map[new_pos[0], new_pos[1]] == 0:
                self.explored_map[new_pos[0], new_pos[1]] = 1
                newly_explored[agent] = True

        # Robot-robot çarpışma kontrolü
        position_counts = {}
        for agent, pos in new_positions.items():
            pos_tuple = tuple(pos)
            if pos_tuple not in position_counts:
                position_counts[pos_tuple] = []
            position_counts[pos_tuple].append(agent)

        # Çarpışan robotları eski pozisyonlarına geri al
        for pos, agents_at_pos in position_counts.items():
            if len(agents_at_pos) > 1:
                for agent in agents_at_pos:
                    collisions[agent] = True
                    new_positions[agent] = self.agent_positions[agent]

        # Pozisyonları güncelle
        self.agent_positions = new_positions

        # Rewards hesapla
        rewards = self._calculate_rewards(newly_explored, collisions)

        # Termination kontrol
        terminations = {agent: False for agent in self.agents}
        truncations = {
            agent: self.step_count >= self.max_steps for agent in self.agents
        }

        # Observations
        observations = {agent: self._get_observation(agent) for agent in self.agents}

        # Info
        exploration_rate = np.sum(self.explored_map) / (
            self.grid_size**2 - self.num_obstacles
        )
        infos = {
            agent: {
                "exploration_rate": exploration_rate,
                "step": self.step_count,
            }
            for agent in self.agents
        }

        return observations, rewards, terminations, truncations, infos

    def _apply_action(self, position: np.ndarray, action: int) -> np.ndarray:
        """Action'ı uygula ve yeni pozisyon döndür"""
        new_pos = position.copy()

        if action == 0:  # Up
            new_pos[0] -= 1
        elif action == 1:  # Down
            new_pos[0] += 1
        elif action == 2:  # Left
            new_pos[1] -= 1
        elif action == 3:  # Right
            new_pos[1] += 1
        # action == 4: Stay (değişiklik yok)

        return new_pos

    def _get_observation(self, agent: str) -> np.ndarray:
        """Agent için observation oluştur"""
        pos = self.agent_positions[agent]

        # Local view oluştur
        local_view = self._get_local_view(pos)

        # Normalized position
        norm_pos = pos / self.grid_size

        # Komşu robotların bilgileri (communication range içinde)
        neighbor_info = self._get_neighbor_info(agent, pos)

        # Tümünü birleştir
        observation = np.concatenate(
            [
                local_view.flatten(),
                norm_pos,
                neighbor_info.flatten(),
            ]
        )

        return observation.astype(np.float32)

    def _get_local_view(self, pos: np.ndarray) -> np.ndarray:
        """Agent'ın local view'ını döndür"""
        view_size = 2 * self.view_range + 1
        local_view = np.zeros((view_size, view_size, 3), dtype=np.float32)

        for i in range(view_size):
            for j in range(view_size):
                world_x = pos[0] - self.view_range + i
                world_y = pos[1] - self.view_range + j

                # Grid sınırları içinde mi?
                if 0 <= world_x < self.grid_size and 0 <= world_y < self.grid_size:
                    # Channel 0: Obstacles
                    local_view[i, j, 0] = self.grid[world_x, world_y]
                    # Channel 1: Explored
                    local_view[i, j, 1] = self.explored_map[world_x, world_y]
                    # Channel 2: Other robots
                    for other_agent, other_pos in self.agent_positions.items():
                        if np.array_equal([world_x, world_y], other_pos):
                            local_view[i, j, 2] = 1.0
                            break

        return local_view

    def _get_neighbor_info(self, agent: str, pos: np.ndarray) -> np.ndarray:
        """Communication range içindeki komşu robotların bilgisi"""
        neighbor_info = np.zeros((self._num_agents - 1, 3), dtype=np.float32)

        idx = 0
        for other_agent in self.agents:
            if other_agent == agent:
                continue

            other_pos = self.agent_positions[other_agent]
            distance = np.linalg.norm(pos - other_pos)

            if distance <= self.communication_range:
                # [relative_x, relative_y, distance_normalized]
                relative = (other_pos - pos) / self.grid_size
                neighbor_info[idx] = [
                    relative[0],
                    relative[1],
                    distance / self.communication_range,
                ]

            idx += 1

        return neighbor_info

    def _calculate_rewards(
        self, newly_explored: Dict[str, bool], collisions: Dict[str, bool]
    ) -> Dict[str, float]:
        """Rewards hesapla"""
        rewards = {}

        # Toplam yeni keşif
        total_new_exploration = sum(newly_explored.values())

        for agent in self.agents:
            reward = 0.0

            # Individual exploration reward
            if newly_explored[agent]:
                reward += self.rewards_config["exploration"]
            else:
                # Redundant exploration penalty
                reward += self.rewards_config["redundant_exploration"]

            # Collision penalty
            if collisions[agent]:
                reward += self.rewards_config["collision"]

            # Step penalty (efficiency)
            reward += self.rewards_config["step"]

            # Collective bonus (tüm robotlar keşif yaptıysa)
            if total_new_exploration >= self._num_agents * 0.6:
                reward += self.rewards_config["collective_bonus"]

            rewards[agent] = reward

        return rewards

    def render(self):
        """Pygame ile görselleştirme"""
        if self.render_mode is None:
            return

        # Calculate window size
        window_size = self.grid_size * self.cell_size

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((window_size, window_size))
            pygame.display.set_caption("Multi-Agent Grid Exploration")
            self.clock = pygame.time.Clock()

        # Background
        self.window.fill((255, 255, 255))

        # Grid çiz
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(
                    y * self.cell_size,
                    x * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )

                # Obstacles
                if self.grid[x, y] == 1:
                    pygame.draw.rect(self.window, (50, 50, 50), rect)
                # Explored
                elif self.explored_map[x, y] == 1:
                    pygame.draw.rect(self.window, (180, 220, 180), rect)
                # Unexplored
                else:
                    pygame.draw.rect(self.window, (240, 240, 240), rect)

                # Grid lines
                pygame.draw.rect(self.window, (200, 200, 200), rect, 1)

        # Robotları çiz
        colors = [
            (70, 130, 180),  # Steel blue
            (220, 20, 60),  # Crimson
            (255, 165, 0),  # Orange
            (147, 112, 219),  # Purple
            (34, 139, 34),  # Forest green
        ]

        for idx, (agent, pos) in enumerate(self.agent_positions.items()):
            center = (
                int(pos[1] * self.cell_size + self.cell_size // 2),
                int(pos[0] * self.cell_size + self.cell_size // 2),
            )
            pygame.draw.circle(
                self.window, colors[idx % len(colors)], center, self.cell_size // 3
            )

            # Communication range çiz (hafif)
            comm_surface = pygame.Surface((window_size, window_size), pygame.SRCALPHA)
            pygame.draw.circle(
                comm_surface,
                (200, 200, 255, 30),
                center,
                int(self.communication_range * self.cell_size),
            )
            self.window.blit(comm_surface, (0, 0))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def close(self):
        """Cleanup"""
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None
