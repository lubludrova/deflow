"""MetaWorld manipulation wrappers with explicit reward and termination rules."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

try:
    import metaworld
except ImportError as exc:
    raise ImportError(
        "metaworld_soft_env requires metaworld==3.1.1 with "
        "gymnasium==1.3.x and mujoco==3.10.x"
    ) from exc


TASKS = {
    "MetaWorldSoftPegInsertSide-v0": "peg-insert-side-v3",
    "MetaWorldSoftButtonPressWall-v0": "button-press-wall-v3",
    "MetaWorldSoftPushWall-v0": "push-wall-v3",
}
OBS_DIM = 39
ACT_DIM = 4
MAX_STEPS = 200
SUCCESS_BONUS = 100.0


class MetaWorldSoftEnv(gym.Env):
    metadata = {"render_modes": []}
    TASK: str | None = None

    def __init__(self, construction_seed: int | None = None, render_mode=None):
        super().__init__()
        if construction_seed is None:
            raise ValueError(
                "construction_seed is required because MetaWorld fixes the MT1 task "
                "cycle when the inner environment is created"
            )
        if self.TASK is None:
            raise ValueError(
                "MetaWorldSoftEnv must be registered through the module-level TASKS table"
            )
        self.construction_seed = int(construction_seed)
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, (ACT_DIM,), dtype=np.float32
        )
        self.env = None
        self.elapsed_steps = 0

    def _build(self) -> None:
        self.env = gym.make(
            "Meta-World/MT1",
            env_name=self.TASK,
            seed=self.construction_seed,
            disable_env_checker=True,
        )
        if self.env.observation_space.shape != (OBS_DIM,):
            raise RuntimeError(
                f"MetaWorld observation contract drifted: {self.env.observation_space.shape}"
            )
        if self.env.action_space.shape != (ACT_DIM,):
            raise RuntimeError(
                f"MetaWorld action contract drifted: {self.env.action_space.shape}"
            )
        if not np.allclose(self.env.action_space.low, -1.0) or not np.allclose(
            self.env.action_space.high, 1.0
        ):
            raise RuntimeError("MetaWorld action bounds drifted from [-1, 1]")

    def reset(self, *, seed=None, options=None):


        if self.env is None:
            self._build()
        observation, info = self.env.reset(seed=seed, options=options)
        self.elapsed_steps = 0
        info = dict(info)
        info.pop("episode", None)
        return np.asarray(observation, dtype=np.float32), info

    def step(self, action):
        observation, dense_reward, terminated, truncated, info = self.env.step(
            np.asarray(action, dtype=np.float32)
        )
        self.elapsed_steps += 1
        info = dict(info)
        info.pop("episode", None)
        success = bool(info.get("success", 0.0))
        shaped_reward = float(dense_reward) + (SUCCESS_BONUS if success else 0.0)
        terminated = bool(terminated) or success
        truncated = bool(truncated) or self.elapsed_steps >= MAX_STEPS
        info["sacflow_dense_reward"] = float(dense_reward)
        info["sacflow_success"] = success
        return (
            np.asarray(observation, dtype=np.float32),
            shaped_reward,
            terminated,
            truncated,
            info,
        )

    def close(self):
        if self.env is not None:
            self.env.close()


def _register(env_id: str, task: str) -> None:
    class MetaWorldSoftTask(MetaWorldSoftEnv):
        TASK = task

    MetaWorldSoftTask.__name__ = f"MetaWorldSoft_{task}".replace("-", "_")
    gym.register(
        id=env_id,
        entry_point=MetaWorldSoftTask,
        max_episode_steps=MAX_STEPS,
    )


for _env_id, _task in TASKS.items():
    _register(_env_id, _task)
