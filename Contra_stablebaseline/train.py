import retro
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import os 
from gymnasium.wrappers import GrayscaleObservation
from gymnasium.wrappers import ResizeObservation

Game = "Contra"

CHECKPOINT_DIR = os.path.join("checkpoint", Game)
TENSORBOARD_DIR = os.path.join("tensorboard", Game)


def make_contra_env():
    env = retro.make(game='Contra-Nes')
    env.render = lambda *args, **kwargs: None
    env = ResizeObservation(env, shape=(84, 84))
    env = GrayscaleObservation(env, keep_dim=True)
    return env

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
