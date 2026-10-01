"""Four-route MultiGoal navigation followed by one precise docking action."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MultiGoalDockingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None, docking_sigma=0.06, navigation_width=1.0):
        super().__init__()
        if docking_sigma <= 0:
            raise ValueError("docking_sigma must be positive")
        if not 0 < navigation_width <= 1:
            raise ValueError("navigation_width must lie in (0, 1]")
        self.docking_sigma = float(docking_sigma)
        self.navigation_width = float(navigation_width)
        self.goals = np.array([[5, 0], [-5, 0], [0, 5], [0, -5]], dtype=np.float32)

        self.docking_targets = np.array(
            [[0.6, 0.2], [-0.6, -0.2], [-0.2, 0.6], [0.2, -0.6]],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([-7, -7, 0, -5, -5], dtype=np.float32),
            high=np.array([7, 7, 1, 5, 5], dtype=np.float32),
        )
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.max_navigation_steps = 30
        self.reach_radius = 1.0

    def _observation(self):
        goal = self.goals[self.selected_goal] if self.selected_goal >= 0 else np.zeros(2, np.float32)
        return np.concatenate((self.pos, [float(self.phase == "dock")], goal)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        fixed_origin = bool((options or {}).get("fixed_origin", False))
        self.pos = (
            np.zeros(2, dtype=np.float32)
            if fixed_origin else self.np_random.uniform(-0.5, 0.5, size=2).astype(np.float32)
        )
        self.phase = "navigate"
        self.selected_goal = -1
        self.navigation_steps = 0
        return self._observation(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if self.phase == "dock":
            distance = float(np.linalg.norm(action - self.docking_targets[self.selected_goal]))
            success = distance <= 2.0 * self.docking_sigma
            self.phase = "done"
            return self._observation(), -distance + 10.0 * success, True, False, {
                "success": float(success),
                "selected_goal": self.selected_goal,
                "docking_distance": distance,
                "docking_hit": float(success),
            }
        if self.phase != "navigate":
            raise RuntimeError("step() called after episode ended")

        self.pos = np.clip(self.pos + action, -7.0, 7.0).astype(np.float32)
        self.navigation_steps += 1
        distances = np.linalg.norm(self.goals - self.pos, axis=1)
        goal = int(np.argmin(distances))
        if self.navigation_width == 1.0:
            reached = bool(distances[goal] < self.reach_radius)
        else:
            offset = self.pos - self.goals[goal]
            direction = self.goals[goal] / 5.0
            parallel = float(offset @ direction)
            transverse = float(offset @ np.array([-direction[1], direction[0]]))
            reached = parallel ** 2 + (transverse / self.navigation_width) ** 2 < 1.0
        if reached:
            self.selected_goal = goal
            self.phase = "dock"
        truncated = not reached and self.navigation_steps >= self.max_navigation_steps
        return self._observation(), -float(distances[goal]), False, truncated, {
            "success": 0.0,
            "selected_goal": self.selected_goal,
            "goal_reached": float(reached),
        }


gym.register(
    id="MultiGoalDock-v0",
    entry_point="multigoal_docking_env:MultiGoalDockingEnv",
    kwargs={"docking_sigma": 0.06},
)
gym.register(
    id="MultiGoalDockTight-v0",
    entry_point="multigoal_docking_env:MultiGoalDockingEnv",
    kwargs={"docking_sigma": 0.03},
)

gym.register(
    id="MultiGoalDockNarrow-v0",
    entry_point="multigoal_docking_env:MultiGoalDockingEnv",
    kwargs={"docking_sigma": 0.03, "navigation_width": 0.20},
)


gym.register(
    id="MultiGoalDockNarrowWide-v0",
    entry_point="multigoal_docking_env:MultiGoalDockingEnv",
    kwargs={"docking_sigma": 0.06, "navigation_width": 0.20},
)
