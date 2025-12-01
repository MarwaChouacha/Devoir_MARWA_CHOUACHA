import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=5, render_mode=None):
        super().__init__()

        self.grid_size = grid_size
        self.render_mode = render_mode

        # Agent starts at (0,0)
        self.agent_pos = np.array([0, 0])

        # Fixed goal
        self.goal_pos = np.array([4, 4])

        # Two obstacles
        self.obstacles = [np.array([1, 2]), np.array([3, 1])]

        # Action space (Up, Down, Left, Right)
        self.action_space = spaces.Discrete(4)

        # Observation: agent position
        self.observation_space = spaces.Box(
            low=0, high=grid_size - 1, shape=(2,), dtype=np.int32
        )

        # Matplotlib figure
        self.fig = None
        self.ax = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_pos = np.array([0, 0])  # reset to start
        return self.agent_pos.copy(), {}

    def step(self, action):
        # Define movements
        moves = {
            0: np.array([-1, 0]),  # up
            1: np.array([1, 0]),   # down
            2: np.array([0, -1]),  # left
            3: np.array([0, 1])    # right
        }

        # Compute next position
        next_pos = self.agent_pos + moves[action]

        # Stay inside boundaries
        next_pos = np.clip(next_pos, 0, self.grid_size - 1)

        # Check obstacle
        if any(np.array_equal(next_pos, obs) for obs in self.obstacles):
            reward = -5
            terminated = False
        elif np.array_equal(next_pos, self.goal_pos):
            self.agent_pos = next_pos
            reward = +10
            terminated = True
        else:
            self.agent_pos = next_pos
            reward = -1
            terminated = False

        return self.agent_pos.copy(), reward, terminated, False, {}

    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(5, 5))

        self.ax.clear()
        self.ax.set_title("GridWorld")

        # Grid
        self.ax.set_xticks(np.arange(self.grid_size+1)-0.5)
        self.ax.set_yticks(np.arange(self.grid_size+1)-0.5)
        self.ax.grid(True)

        # Draw obstacles
        for obs in self.obstacles:
            self.ax.add_patch(plt.Rectangle(obs[::-1] - 0.5, 1, 1, color="black"))

        # Draw goal
        g = self.goal_pos
        self.ax.add_patch(plt.Rectangle(g[::-1] - 0.5, 1, 1, color="green"))

        # Draw agent
        a = self.agent_pos
        self.ax.add_patch(plt.Rectangle(a[::-1] - 0.5, 1, 1, color="blue"))

        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_aspect("equal")

        plt.pause(0.01)
        plt.draw()
