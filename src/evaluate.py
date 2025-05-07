"""Deterministic evaluator using argmax policy."""
from __future__ import annotations

import pathlib

import torch
import torch.nn.functional as F

from .config import TrainingConfig as Cfg
from .envs import make_env
from .model import PPOAgent
import pathlib
import time


def evaluate(ckpt: str, episodes: int = 5, skip = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_dir = pathlib.Path("evaluation_videos")
    video_dir.mkdir(exist_ok=True)

    
    # Build a dummy env to get obs shape (with same skip)
    dummy_env = make_env(Cfg.LEVEL, skip=skip)
    obs_channels = dummy_env.observation_space.shape[0]
    dummy_env.close()

    agent = PPOAgent(obs_channels, 8).to(device)
    agent.load_state_dict(torch.load(ckpt, map_location=device))
    agent.eval()

    successes = 0
    for ep in range(episodes):
        video_file = video_dir / f"eval_ep{ep+1}_{int(time.time())}.mp4"
        env = make_env(Cfg.LEVEL, video_path=str(video_file), skip=skip)
        # print(len(env._mapping)) 
        # print(env._mapping[7])
        state = torch.from_numpy(env.reset()).to(device)
        done, info = False, {}
        while not done:
            with torch.no_grad():
                logits, _ = agent(state)
            action = torch.argmax(F.softmax(logits, dim=-1)).item()
            state_np, _, done, info = env.step(action)
            state = torch.from_numpy(state_np).to(device)

        env.close() 

        outcome = "SUCCESS" if info.get("lives", 0) > 0 else "FAIL"
        if outcome == "SUCCESS":
            successes += 1
        print(f"Episode {ep+1}: {outcome} → saved to {video_file}")

    print(f"Win rate: {successes/episodes:.2%}")

if __name__ == "__main__":
    evaluate("trained_models/ppo_contra_deterministic.pth", episodes=1)
