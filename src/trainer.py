"""PPO training loop (single‑process learner + vectorized envs)."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .config import TrainingConfig as Cfg
from .envs import ParallelEnvs
from .model import PPOAgent
from .envs import make_env


class PPOTrainer:
    """Implements clipped‑objective PPO with GAE."""

    def __init__(self, cfg: Cfg):
        self._cfg = cfg
        self._envs = ParallelEnvs(cfg.LEVEL, cfg.NUM_ENVS)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._agent = PPOAgent(self._envs.num_states, self._envs.num_actions).to(self._device)
        self._opt = torch.optim.Adam(self._agent.parameters(), lr=cfg.LEARNING_RATE)

        # Logging dirs
        Path(cfg.CKPT_DIR).mkdir(exist_ok=True)
        if os.path.isdir(cfg.LOG_DIR):
            shutil.rmtree(cfg.LOG_DIR)
        os.makedirs(cfg.LOG_DIR)

    def _collect_rollout(self):
        states = torch.from_numpy(self._envs.reset()).to(self._device)
        batch_states, batch_actions, batch_rewards, batch_dones, batch_logp, batch_values = [], [], [], [], [], []
        for _ in range(self._cfg.ROLLOUT_STEPS):
            logits, values = self._agent(states)
            probs = F.softmax(logits, dim=-1)
            m = Categorical(probs)
            actions = m.sample()

            next_states_np, rewards_np, dones_np, _ = self._envs.step(actions.cpu().numpy())

            batch_states.append(states)
            batch_actions.append(actions)
            batch_rewards.append(torch.tensor(rewards_np, device=self._device, dtype=torch.float32))
            batch_dones.append(torch.tensor(dones_np.astype(np.float32), device=self._device))
            batch_logp.append(m.log_prob(actions))
            batch_values.append(values.squeeze())

            states = torch.from_numpy(next_states_np).to(self._device)
        return map(torch.cat, (batch_states, batch_actions, batch_rewards, batch_dones, batch_logp, batch_values)), states

    def _compute_returns(self, rewards, dones, last_values):
        returns, gae = [], 0.0
        for t in reversed(range(len(rewards))):      # loop T-1 .. 0
            delta = (
                rewards[t]
                + self._cfg.GAMMA * last_values[t + 1] * (1.0 - dones[t])
                - last_values[t]
            )
            gae = delta + self._cfg.GAMMA * self._cfg.GAE_TAU * (1.0 - dones[t]) * gae
            returns.insert(0, gae + last_values[t])
        return torch.stack(returns)

    def train(self):
        episode = 0
        while True:
            (
                states,
                actions,
                rewards,
                dones,
                old_logp,
                values,
            ), last_states = self._collect_rollout()

            with torch.no_grad():
                _, last_values = self._agent(last_states)
            returns = self._compute_returns(rewards, dones, torch.cat((values, last_values.squeeze())))
            advantages = returns - values

            states     = states.detach()
            values     = values.detach()
            old_logp   = old_logp.detach()
            returns    = returns.detach()
            advantages = advantages.detach()

            total_loss = 0.0
            loss_count = 0

            for _ in range(self._cfg.NUM_EPOCHS):
                indices = torch.randperm(len(states))
                for start in range(0, len(states), self._cfg.BATCH_SIZE):
                    idx = indices[start : start + self._cfg.BATCH_SIZE]
                    logits, value = self._agent(states[idx])
                    dist = Categorical(F.softmax(logits, dim=-1))
                    ratio = torch.exp(dist.log_prob(actions[idx]) - old_logp[idx])
                    actor_loss = -torch.mean(
                        torch.min(
                            ratio * advantages[idx],
                            torch.clamp(ratio, 1.0 - self._cfg.CLIP_EPS, 1.0 + self._cfg.CLIP_EPS) * advantages[idx],
                        )
                    )
                    critic_loss = F.smooth_l1_loss(returns[idx], value.squeeze())
                    entropy = torch.mean(dist.entropy())
                    loss = actor_loss + critic_loss - self._cfg.ENTROPY_BETA * entropy

                    self._opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._agent.parameters(), 0.5)
                    self._opt.step()

                    total_loss += loss.item()
                    loss_count += 1

            avg_loss = total_loss / loss_count if loss_count > 0 else 0.0
            print(f"Episode {episode}: avg_loss={avg_loss:.4f}")

            episode += 1
            if episode % self._cfg.SAVE_EVERY_EPISODES == 0:
                ckpt_path = os.path.join(self._cfg.CKPT_DIR, f"ppo_contra_ep{episode}.pt")
                torch.save(self._agent.state_dict(), ckpt_path)
                print("Saved checkpoint", ckpt_path)
            
            if episode % 50 == 0: 
                win = self._quick_eval()
                if win:
                    ckpt = f"{self._cfg.CKPT_DIR}/contra_win_ep{episode}.pt"
                    torch.save(self._agent.state_dict(), ckpt)
                    print("Win, save checkpoint", ckpt)

    def _quick_eval(self) -> bool:
        env = make_env(self._cfg.LEVEL)
        state = torch.from_numpy(env.reset()).to(self._device)
        done, info = False, {}
        while not done:
            with torch.no_grad():
                logits, _ = self._agent(state)
                act = torch.argmax(F.softmax(logits, -1)).item()
            state_np, _, done, info = env.step(act)
            state = torch.from_numpy(state_np).to(self._device)
        return info.get("lives", 0) > 0


