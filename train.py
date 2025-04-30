import os, sys
import retro
import numpy as np
import torch
import torch.optim as optim
import cv2

# ensure our package folder is on sys.path when running from project_root
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "contraforce_ppo")))

from contraforce_torch.config import *
from contraforce_torch.agent  import PPOAgent
from contraforce_torch.buffer import RolloutBuffer, compute_gae
from contraforce_torch.utils  import preprocess, init_stack, stack_frames

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ppo_update(policy, optimizer, buffer: RolloutBuffer, step: int):
    """Perform PPO update using collected rollouts."""
    returns   = compute_gae(buffer.next_value, buffer.rewards, buffer.dones, buffer.values)
    states    = torch.stack(buffer.states).to(device)
    actions   = torch.tensor(buffer.actions).to(device)
    old_logps = torch.tensor(buffer.logprobs).to(device)
    returns_t = torch.tensor(returns).to(device)
    values_t  = torch.tensor(buffer.values).to(device)
    advs      = returns_t - values_t
    advs      = (advs - advs.mean()) / (advs.std() + 1e-8)

    for _ in range(PPO_EPOCHS):
        idxs = np.random.permutation(len(states))
        for start in range(0, len(states), BATCH_SIZE):
            batch = idxs[start:start + BATCH_SIZE]
            b_s   = states[batch]
            b_a   = actions[batch]
            b_old = old_logps[batch]
            b_ret = returns_t[batch]
            b_adv = advs[batch]

            logits, vals = policy(b_s)
            dist         = torch.distributions.Categorical(logits=logits)
            new_logps    = dist.log_prob(b_a)
            entropy      = dist.entropy().mean()

            ratio = (new_logps - b_old).exp()
            s1    = ratio * b_adv
            s2    = torch.clamp(ratio, 1 - CLIP_PARAM, 1 + CLIP_PARAM) * b_adv

            actor_loss  = -torch.min(s1, s2).mean()
            critic_loss = (b_ret - vals.squeeze()).pow(2).mean()
            loss        = actor_loss + 0.5 * critic_loss - ENTROPY_BONUS * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"[Update {step}] Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")

def train():
    env = retro.make(game="ContraForce-Nes")
    action_dim = env.action_space.shape[0]

    policy    = PPOAgent(action_dim, in_channels=STACK_SIZE).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    buffer    = RolloutBuffer()

    obs         = env.reset()
    frame       = preprocess(obs)
    stacked     = init_stack(frame)
    state_arr   = np.expand_dims(np.stack(stacked), axis=0)
    state_tensor= torch.tensor(state_arr, device=device)

    prev_score  = 0
    prev_obs    = obs.copy()
    episode_r   = 0
    update_step = 0
    ep_count    = 0

    for t in range(1, MAX_TIMESTEPS + 1):
        if RENDER:
            cv2.imshow("ContraForce", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        with torch.no_grad():
            action, logp, _, value = policy.get_action(state_tensor)
        act_arr = np.zeros(action_dim, dtype=np.int8)
        act_arr[action.item()] = 1

        repeat = JUMP_REPEAT if act_arr[8] == 1 else DEFAULT_REPEAT
        total_reward = 0.0
        done = False

        for _ in range(repeat):
            next_obs, _, done, info = env.step(act_arr)
            # score-based reward
            s = info.get("score", 0)
            r = (s - prev_score) * SCORE_SCALE
            prev_score = s
            # pixel-change reward on right half
            pr = prev_obs[:, prev_obs.shape[1]//2:]
            cr = next_obs[:, next_obs.shape[1]//2:]
            diff = np.abs(cr.astype(np.float32) - pr.astype(np.float32)).mean()
            r += diff * PIXEL_SCALE
            # death penalty
            if done and info.get("lives", 0) <= 0:
                r -= 10.0
            # unnecessary jump
            if act_arr[8] == 1 and diff < 0.01:
                r -= JUMP_PENALTY

            total_reward += r
            prev_obs = next_obs.copy()
            if done:
                break

        new_frame = preprocess(next_obs)
        stacked, stacked_arr = stack_frames(stacked, new_frame)
        next_state = torch.tensor(
            np.expand_dims(stacked_arr, 0), device=device
        )

        # store in buffer
        buffer.states.append(state_tensor.squeeze(0))
        buffer.actions.append(action)
        buffer.logprobs.append(logp.cpu())
        buffer.rewards.append(total_reward)
        buffer.dones.append(1 - int(done))
        buffer.values.append(value.squeeze(0).cpu().item())

        state_tensor = next_state
        obs = next_obs
        episode_r += total_reward

        if t % UPDATE_INTERVAL == 0:
            with torch.no_grad():
                _, buffer.next_value = policy(state_tensor)
                buffer.next_value = buffer.next_value.squeeze(0).cpu().item()

            update_step += 1
            ppo_update(policy, optimizer, buffer, update_step)
            buffer.clear()

        if done:
            ep_count += 1
            print(f"[Episode {ep_count}] Reward: {episode_r:.2f}, Score: {s}, Lives: {info.get('lives')}")
            # reset
            episode_r = 0
            obs       = env.reset()
            frame     = preprocess(obs)
            stacked   = init_stack(frame)
            state_arr = np.expand_dims(np.stack(stacked), axis=0)
            state_tensor = torch.tensor(state_arr, device=device)
            prev_score  = 0
            prev_obs    = obs.copy()

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    train()
