"""
Project‑wide constants and hyperparameters.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:

    RAND_EVERY: int = 50
    RAND_STEPS: int = 0

    VIDEO_WIDTH: int = 480
    VIDEO_HEIGHT: int = 448
    VIDEO_BITRATE: str = "4M"

    # Environment
    LEVEL: int = 1
    NUM_ENVS: int = 32

    # PPO hyper‑parameters
    LEARNING_RATE: float = 1e-4
    GAMMA: float = 0.9
    GAE_TAU: float = 1.0
    ENTROPY_BETA: float = 0.01
    CLIP_EPS: float = 0.2

    ROLLOUT_STEPS: int = 128
    NUM_EPOCHS: int = 10
    BATCH_SIZE: int = 128

    # Runtime
    SAVE_EVERY_EPISODES: int = 200
    MAX_EPISODE_STEPS: int = 10_000

    # Paths
    LOG_DIR: str = "tensorboard/ppo_contra"
    CKPT_DIR: str = "trained_models"