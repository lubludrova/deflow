"""ManiSkill PickCube and PushCube wrappers for the shared training loop."""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np

try:
    import mani_skill
    _MANI_SKILL_AVAILABLE = True
except ImportError:
    _MANI_SKILL_AVAILABLE = False


if _MANI_SKILL_AVAILABLE:
    try:
        from mani_skill.envs.utils.system import backend as _maniskill_backend
        _maniskill_backend.render_backend_name_mapping["sacflow_pci0"] = "pci:0"
    except (ImportError, AttributeError):
        pass


def _require_mani_skill() -> None:


    if not _MANI_SKILL_AVAILABLE:
        try:
            import mani_skill
        except ImportError as e:
            raise ImportError(
                "mani_skill is required for MS3* envs "
                "(pip install mani-skill==3.0.x; tested on 3.0.1; "
                "GPU PhysX is NOT used by this wrapper — CPU only)."
            ) from e
        try:
            from mani_skill.envs.utils.system import backend
            backend.render_backend_name_mapping["sacflow_pci0"] = "pci:0"
        except (ImportError, AttributeError):
            pass


TASK_REGISTRY: dict[str, dict] = {
    "PickCube-v1": {
        "short": "pickcube",
        "role": "Easier pick-and-place manipulation control",
        "robot": "panda",
        "default_obs_mode": "state",
        "default_control_mode": "pd_joint_delta_pos",
        "default_reward_mode": "dense",
        "max_episode_steps": 50,
        "default_render_mode": "rgb_array",
        "default_num_envs": 1,
        "has_dense_reward": True,
        "task_kind": "training",
    },
    "PushCube-v1": {
        "short": "pushcube",
        "role": "Non-prehensile push into a goal region",
        "robot": "panda",
        "default_obs_mode": "state",
        "default_control_mode": "pd_joint_delta_pos",
        "default_reward_mode": "dense",
        "max_episode_steps": 50,
        "default_render_mode": "rgb_array",
        "default_num_envs": 1,
        "has_dense_reward": True,
        "task_kind": "training",
    },
}


class ManiSkillFlat(gym.Env):


    metadata = {"render_modes": ["rgb_array", "human"]}

    SUCCESS_BONUS = 1.0

    def __init__(
        self,
        task_id: str,
        max_episode_steps: int | None = None,
        obs_mode: str = "state",
        control_mode: str | None = None,
        reward_mode: str | None = None,
        render_mode: str | None = None,
        sim_backend: str = "cpu",
        render_backend: str | None = "none",
        robot: str | None = None,
    ):
        super().__init__()
        _require_mani_skill()
        cfg = TASK_REGISTRY.get(task_id)
        if cfg is None:
            raise ValueError(
                f"Unknown ManiSkill task '{task_id}'. Known: "
                f"{sorted(TASK_REGISTRY.keys())}"
            )
        self.task_id = task_id
        self.task_cfg = cfg
        self.obs_mode = obs_mode
        self.control_mode = control_mode
        self.reward_mode = reward_mode or cfg["default_reward_mode"]
        self.render_mode = render_mode
        self.sim_backend = sim_backend
        self.render_backend = render_backend
        self.robot = robot or cfg["robot"]
        self._max_episode_steps = max_episode_steps or cfg["max_episode_steps"]

        kwargs: dict = dict(
            obs_mode=obs_mode,
            max_episode_steps=self._max_episode_steps,
            sim_backend=sim_backend,
            render_backend=render_backend,
            robot_uids=self.robot,
        )
        if control_mode is not None:
            kwargs["control_mode"] = control_mode
        if reward_mode is not None:
            kwargs["reward_mode"] = reward_mode

        if render_mode is not None:
            kwargs["render_mode"] = render_mode

        self.env = gym.make(task_id, **kwargs)
        space = self.env.observation_space
        if isinstance(space, gym.spaces.Box):
            self._keys = None
            low, high = space.low, space.high
            if len(space.shape) > 1 and space.shape[0] == 1:
                low, high = low[0], high[0]
            self.observation_space = gym.spaces.Box(
                low=low, high=high, dtype=np.float32
            )
        elif hasattr(space, "spaces"):


            self._keys = sorted(space.spaces.keys())
            dim = int(
                sum(int(np.prod(space.spaces[k].shape)) for k in self._keys)
            )
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32
            )
        else:
            raise ValueError(
                f"obs_mode={obs_mode!r} on {task_id} did not return a Dict or "
                f"Box obs space (got {type(space).__name__}); "
                f"use obs_mode='state'."
            )
        self.action_space = self.env.action_space


    def _flat(self, o):
        if self._keys is None:
            if hasattr(o, "detach"):
                o = o.detach().cpu().numpy()
            return np.asarray(o, dtype=np.float32).reshape(-1)
        return np.concatenate(
            [np.asarray(o[k], dtype=np.float32).reshape(-1) for k in self._keys]
        )

    def _normalize_action(self, action):


        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()
        return np.asarray(action, dtype=np.float32)


    def reset(self, *, seed=None, options=None):
        o, info = self.env.reset(seed=seed, options=options)
        info.pop("episode", None)
        return self._flat(o), info

    def step(self, action):
        o, r, term, trunc, info = self.env.step(self._normalize_action(action))
        info.pop("episode", None)
        success = info.get("success", False)
        if hasattr(success, "item"):
            success = success.item()
        success = bool(success)
        if success:
            r = float(r) + self.SUCCESS_BONUS
            term = True
        return self._flat(o), float(r), term, trunc, info

    def render(self):
        return self.env.render()

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


    def get_state(self) -> np.ndarray:


        state = self.env.unwrapped.get_state()
        return state.detach().cpu().numpy().astype(np.float64, copy=True)

    def set_state(self, state) -> None:

        import torch

        inner = self.env.unwrapped
        reference = inner.get_state()
        inner.set_state(torch.as_tensor(
            state, dtype=reference.dtype, device=reference.device
        ))

    def get_full_state(self) -> dict[str, np.ndarray]:


        payload = {}

        def flatten(node, prefix=""):
            for key, value in node.items():
                path = f"{prefix}/{key}" if prefix else key
                if isinstance(value, dict):
                    flatten(value, path)
                else:
                    if hasattr(value, "detach"):
                        value = value.detach().cpu().numpy()
                    payload[path] = np.asarray(value).copy()

        flatten(self.env.unwrapped.get_state_dict())
        return payload

    def set_full_state(self, payload) -> None:


        import torch

        inner = self.env.unwrapped
        used = set()

        def restore(node, prefix=""):
            restored = {}
            for key, reference in node.items():
                path = f"{prefix}/{key}" if prefix else key
                if isinstance(reference, dict):
                    restored[key] = restore(reference, path)
                    continue
                if path not in payload:
                    raise ValueError(f"Full-state payload missing {path!r}")
                used.add(path)
                value = np.asarray(payload[path])
                if value.shape != tuple(reference.shape):
                    raise ValueError(
                        f"Full-state shape for {path!r}: "
                        f"{value.shape} != {tuple(reference.shape)}"
                    )
                if isinstance(reference, torch.Tensor):
                    restored[key] = torch.as_tensor(
                        value, dtype=reference.dtype, device=reference.device
                    ).clone()
                else:
                    restored[key] = np.asarray(value, dtype=reference.dtype).copy()
            return restored

        restored = restore(inner.get_state_dict())
        if set(payload) != used:
            raise ValueError(f"Unexpected full-state paths: {sorted(set(payload) - used)}")
        inner.set_state_dict(restored)
        if "controller" in restored:
            inner.agent.set_controller_state(restored["controller"])

    def get_info(self):


        return self.env.unwrapped.get_info()


def _make_subclass(task_id: str, **defaults):


    def factory(**kwargs):
        merged = {**defaults, **kwargs}
        return ManiSkillFlat(task_id, **merged)

    factory.__name__ = f"_make_{task_id.replace('-', '_')}"
    return factory


for _tid, _cfg in TASK_REGISTRY.items():


    _short_name, _ = _tid.rsplit("-", 1)
    _cls_name = f"MS3{_short_name}-v0"
    gym.register(
        id=_cls_name,
        entry_point=_make_subclass(
            _tid,
            max_episode_steps=_cfg["max_episode_steps"],
            reward_mode=_cfg["default_reward_mode"],
            render_backend=os.environ.get(
                "SACFLOW_MANISKILL_RENDER_BACKEND", "sapien_cpu"
            ),
        ),
        max_episode_steps=_cfg["max_episode_steps"],
    )


def make_maniskill_env(
    task_id: str,
    *,
    num_envs: int = 1,
    obs_mode: str | None = None,
    control_mode: str | None = None,
    reward_mode: str | None = None,
    render_mode: str | None = None,
    sim_backend: str = "cpu",
    render_backend: str | None = "sapien_cpu",
    seed: int | None = None,
    max_episode_steps: int | None = None,
    device: str | None = None,
    robot: str | None = None,
    record_video: bool = False,
    record_trajectory: bool = False,
    video_dir: str | None = None,
):


    if task_id not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown ManiSkill task '{task_id}'. Known: "
            f"{sorted(TASK_REGISTRY.keys())}"
        )
    if num_envs != 1:
        raise ValueError(
            f"make_maniskill_env only supports num_envs=1; got {num_envs}. "
            f"Use Gymnasium SyncVectorEnv with separate instances for n>1."
        )
    if sim_backend not in ("cpu", "physx_cpu", "physx_cuda", "cuda", "gpu"):
        raise ValueError(
            f"Unsupported sim_backend {sim_backend!r}; expected one of "
            f"'cpu', 'physx_cpu', 'physx_cuda' (or aliases 'cuda'/'gpu')."
        )
    if render_backend is not None and render_backend not in (
        "none", "cpu", "cuda", "gpu", "sapien_cpu", "sapien_cuda",
        "sacflow_pci0",
    ):
        raise ValueError(
            f"Unsupported render_backend {render_backend!r}; expected one of "
            f"'none', 'cpu', 'cuda' (or aliases 'gpu', 'sapien_*', "
            f"'sacflow_pci0')."
        )

    cfg = TASK_REGISTRY[task_id]

    if reward_mode is not None and reward_mode not in ("none", "sparse") \
            and not cfg["has_dense_reward"]:
        raise ValueError(
            f"{task_id} does not provide a dense reward "
            f"(SUPPORTED_REWARD_MODES = {{'none','sparse'}}); "
            f"requested reward_mode={reward_mode!r}. Use 'sparse' or 'none'."
        )

    env = ManiSkillFlat(
        task_id=task_id,
        max_episode_steps=max_episode_steps,
        obs_mode=obs_mode or cfg["default_obs_mode"],
        control_mode=control_mode or cfg["default_control_mode"],
        reward_mode=reward_mode or cfg["default_reward_mode"],
        render_mode=render_mode or cfg["default_render_mode"]
        if render_mode != "rgb_array" else None,
        sim_backend=sim_backend,
        render_backend=render_backend,
        robot=robot or cfg["robot"],
    )
    if seed is not None:
        env.reset(seed=seed)
    return env


def list_tasks() -> list[str]:

    return sorted(TASK_REGISTRY.keys())


def task_info(task_id: str) -> dict:

    if task_id not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown ManiSkill task '{task_id}'. Known: "
            f"{sorted(TASK_REGISTRY.keys())}"
        )
    return dict(TASK_REGISTRY[task_id])
