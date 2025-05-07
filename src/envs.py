"""Environment helpers and custom wrappers for Contra‑Nes"""
from __future__ import annotations

import random
import subprocess as _sp
from typing import List, Tuple

from .config import TrainingConfig as cfg
import cv2
import gymnasium as gym
import numpy as np
import retro
import torch.multiprocessing as mp
from gymnasium.spaces import Box

BUTTON_COMBOS = [
    ("B",),
    ("LEFT",  "B"),
    ("RIGHT", "B"),
    ("DOWN",  "B"), 
    ("A", "B"),
    ("RIGHT", "UP",   "B"),
    ("RIGHT", "DOWN", "B"),
    ("RIGHT", "A", "B"),
]

def _preprocess(frame: np.ndarray | None) -> np.ndarray:
    """Convert RGB frame to (1,84,84) gray, but pass through if already gray."""
    if frame is None:
        return np.zeros((1, 84, 84), dtype=np.float32)

    # Already pre‑processed?
    if frame.ndim == 3 and frame.shape[0] == 1:
        return frame.astype(np.float32)
    if frame.ndim == 2 and frame.shape == (84, 84):
        return frame[None, :, :].astype(np.float32)

    if frame.ndim == 3 and frame.shape[2] in (3, 4):  # HWC RGB
        gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"Unexpected frame shape {frame.shape} in _preprocess")

    gray = cv2.resize(gray, (84, 84))
    return gray[None, :, :].astype(np.float32) / 255.0


def _safe_reset(env, **kwargs):
    """Return only observation, new/old Gym API compatible."""
    res = env.reset(**kwargs)
    return res[0] if isinstance(res, tuple) else res

class RewardWrapper(gym.Wrapper):
    """Adds shaped reward and preprocessing."""

    def __init__(self, env: gym.Env, mapping: List[List[int]], monitor: _Monitor | None = None):
        super().__init__(env)
        self.observation_space = Box(low=0, high=1.0, shape=(1, 84, 84), dtype=np.float32)
        self._mapping = mapping
        self._monitor = monitor
        self._prev_x = 0
        self._prev_score = 0
        self._prev_lives = 2

    def step(self, action_idx):  # type: ignore[override]
        action = action_idx
        if isinstance(action, (list, np.ndarray)):
            mapped_action = action
        else:
            mapped_action = self._mapping[action] 

        state, _, terminated, truncated, info = self.env.step(mapped_action)
        done = terminated or truncated

        if self._monitor:
            self._monitor.record(state)

        _state_idle, *_ = self.env.step([0] * len(self._mapping[0]))
        if self._monitor:
            self._monitor.record(_state_idle)

        state_p = _preprocess(state)
        reward = np.clip(info["xscroll"] - self._prev_x - 0.01, -3, 3)
        reward += np.clip(info["score"] - self._prev_score, 0, 2)
        self._prev_x = info["xscroll"]
        self._prev_score = info["score"]

        if info["lives"] < self._prev_lives:
            reward -= 15
            self._prev_lives = info["lives"]

        if done:
            reward += 50 if info["lives"] > 0 else -35

        return state_p, reward / 10.0, done, info

    def reset(self, **kwargs):
        self._prev_x = self._prev_score = 0
        self._prev_lives = 2
        obs = _safe_reset(self.env, **kwargs)
        return _preprocess(obs)


class SkipFrameWrapper(gym.Wrapper):
    """Skip + max‑pooling over last two frames with optional random actions."""

    def __init__(
        self,
        env: gym.Env,
        mapping: List[List[int]],
        skip: int = 4,
        rand_every: int = 50,
        rand_steps: int = 5,
    ):
        super().__init__(env)
        self.observation_space = Box(low=0, high=1.0, shape=(skip, 84, 84), dtype=np.float32)
        self._mapping = mapping
        self._skip = skip
        self._rand_every = rand_every
        self._rand_steps = rand_steps
        self._frames = np.zeros((skip, 84, 84), dtype=np.float32)
        self._t = 0

    def _step_single(self, mapped_action):
        """Step underlying env; handle both 4‑tuple and 5‑tuple formats."""
        result = self.env.step(mapped_action)
        if len(result) == 5:
            state, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:  # old API or RewardWrapper output
            state, reward, done, info = result
        return state, reward, done, info

    def step(self, action_idx): 
        total_reward, done, info = 0.0, False, {}
        states: list[np.ndarray] = []

        if self._rand_steps > 0 and self._t % self._rand_every == 0 and self._t > 0:
            seq = random.choices(self._mapping, k=self._rand_steps)
        else:
            seq = [self._mapping[action_idx]] * self._skip

        for act in seq:
            state, reward, done, info = self._step_single(act)
            total_reward += reward
            states.append(state)
            if done:
                break

        if states:
            pooled = np.max(np.stack(states[-2:]), axis=0)  # (1,84,84)
            self._frames[:-1] = self._frames[1:]
            self._frames[-1] = pooled[0]

        self._t += 1
        return self._frames[None, :, :, :], total_reward, done, info

    def reset(self, **kwargs):
        self._t = 0
        obs = _safe_reset(self.env, **kwargs)
        # obs is already (1,84,84)
        frame = obs[0]
        self._frames = np.repeat(frame[None, :, :], self._skip, axis=0)
        return self._frames[None, :, :, :]


def make_env(level: int, video_path: str | None = None, skip: int = 0):
    base = retro.make(
        "Contra-Nes",
        state=f"Level{level}",
        use_restricted_actions=retro.Actions.FILTERED,
    )
    base.viewer = None
    base.render = lambda *_, **__: None  # noqa: E731

    buttons = base.unwrapped.buttons  # type: ignore[attr-defined]
    mapping = _make_action_mapping(buttons)

    monitor = _Monitor(240, 224, video_path) if video_path else None
    env: gym.Env = RewardWrapper(base, mapping, monitor)
    env = SkipFrameWrapper(env, mapping, rand_every=50, rand_steps=skip)
    return env


class ParallelEnvs:
    """Minimal vector‑like pool using Python multiprocessing pipes."""

    def __init__(self, level: int, n: int):
        self._parent_conns, self._child_conns = zip(*[mp.Pipe() for _ in range(n)])
        self._procs = []
        for idx in range(n):
            proc = mp.Process(target=self._worker, args=(idx, level))
            proc.daemon = True
            proc.start()
            self._child_conns[idx].close()
            self._procs.append(proc)

        # Probe shapes and mapping once
        env = make_env(level)
        self.num_states = env.observation_space.shape[0]
        self.num_actions = len(env.env._mapping)  # type: ignore[attr-defined]

    def _worker(self, idx: int, level: int):
        env = make_env(level)
        parent = self._child_conns[idx]
        self._parent_conns[idx].close()
        while True:
            cmd, data = parent.recv()
            if cmd == "step":
                parent.send(env.step(int(data)))
            elif cmd == "reset":
                parent.send(_safe_reset(env))
            else:
                raise ValueError(f"Unknown cmd {cmd}")

    # External API
    def reset(self) -> np.ndarray:
        for conn in self._parent_conns:
            conn.send(("reset", None))
        return np.concatenate([conn.recv() for conn in self._parent_conns])

    def step(self, actions) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        for conn, act in zip(self._parent_conns, actions):
            conn.send(("step", act))
        results = [conn.recv() for conn in self._parent_conns]
        states, rewards, dones, infos = zip(*results)
        return np.concatenate(states), np.array(rewards), np.array(dones), list(infos)


class _Monitor:
    def __init__(self, width: int, height: int, path: str) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", "60",
            "-i", "-",
            # --------------------------
            "-vf", f"scale={width*2}:{height*2}",
            "-vcodec", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            path,
        ]
        self._pipe = _sp.Popen(cmd, stdin=_sp.PIPE, stderr=_sp.DEVNULL)

    def record(self, frame: np.ndarray) -> None:
        if self._pipe.stdin:
            self._pipe.stdin.write(frame.tobytes())


def _make_action_mapping(buttons: List[str]) -> List[List[int]]:
    """
    Generate binary action mapping from BUTTON_COMBOS and actual button list.

    Args:
        buttons: list of button names in order from the environment.
    Returns:
        A list of binary arrays, one per combo.
    """
    mapping: List[List[int]] = []
    for combo in BUTTON_COMBOS:
        arr = [1 if btn in combo else 0 for btn in buttons]
        mapping.append(arr)
    return mapping
