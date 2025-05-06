import retro
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import os 
from gymnasium.wrappers import GrayscaleObservation
from gymnasium.wrappers import ResizeObservation
import cv2
import numpy as np

CHECKPOINT_DIR = os.path.join("checkpoint", "Contra")
TENSORBOARD_DIR = os.path.join("tensorboard", "Contra")


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized[:, :, None]

def make_contra_env():
    env = retro.make(game='Contra-Nes')

    # Rewrite render method to skip rendering
    env.render = lambda *args, **kwargs: None

    return PreprocessFrame(env)

vec_env = DummyVecEnv([make_contra_env])
vec_env = VecFrameStack(vec_env, n_stack=4)

print("Obs shape:", vec_env.observation_space.shape)

tensorboard_log = os.path.join(TENSORBOARD_DIR, "ppo_contra_tensorboard")

model = PPO(
    "CnnPolicy",
    vec_env,
    verbose=1,
    device="cuda",
    tensorboard_log=tensorboard_log
)

model.learn(total_timesteps=1_000_000)

model.save(os.path.join(CHECKPOINT_DIR, f"ppo_contra_nes{model.num_timesteps}"))
