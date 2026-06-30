import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.distributions import Normal
import bdh
from policy_models import BDHActorCritic
from env import RandomizedPendulumEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PPO_EPOCHS = 100
EPISODES_PER_BATCH = 10
HORIZON = 200
K_EPOCHS = 4
GAMMA = 0.99
LAMBDA = 0.95
EPS_CLIP = 0.2
LR = 3e-4

def compute_gae(rewards, values, next_value, gamma, lam):
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            delta = rewards[step] + gamma * next_value - values[step]
        else:
            delta = rewards[step] + gamma * values[step + 1] - values[step]
        gae = delta + gamma * lam * gae
        returns.insert(0, gae + values[step])
    return returns

def train_ppo():
    # Unprivileged environment: BDH must infer gravity from history
    env = RandomizedPendulumEnv(privileged=False)
    
    bdh_config = bdh.BDHConfig(
        n_layer=3, n_embd=64, n_head=4, mlp_internal_dim_multiplier=8,
        state_dim=3, action_dim=1
    )
    policy = BDHActorCritic(bdh_config).to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=LR)
    
    print("Starting PPO Training for BDH...")
    
    for epoch in range(PPO_EPOCHS):
        # 1. Collect Rollouts
        policy.eval()
        all_states = []
        all_actions = []
        all_logprobs = []
        all_rewards = []
        all_values = []
        all_returns = []
        all_advantages = []
        
        batch_reward = 0
        
        for _ in range(EPISODES_PER_BATCH):
            state, _ = env.reset()
            
            states = []
            actions = []
            logprobs = []
            rewards = []
            values = []
            
            history_states = torch.tensor([state], dtype=torch.float32, device=device).unsqueeze(0)
            
            for t in range(HORIZON):
                states.append(state)
                
                with torch.no_grad():
                    action_mean, action_log_std, value = policy(history_states)
                    action_mean = action_mean[:, -1, :]
                    value = value[:, -1, :].item()
                    
                    action_std = torch.exp(action_log_std)
                    dist = Normal(action_mean, action_std)
                    action = dist.sample()
                    logprob = dist.log_prob(action).sum(dim=-1).item()
                    
                action_np = action.cpu().numpy()[0]
                action_np = np.clip(action_np, -2.0, 2.0)
                
                next_state, reward, terminated, truncated, _ = env.step(action_np)
                
                actions.append(action_np)
                logprobs.append(logprob)
                rewards.append(reward)
                values.append(value)
                
                state = next_state
                obs_tensor = torch.tensor([state], dtype=torch.float32, device=device).unsqueeze(0)
                history_states = torch.cat([history_states, obs_tensor], dim=1)
                
                if terminated or truncated:
                    break
                    
            batch_reward += sum(rewards)
            
            # Compute GAE
            with torch.no_grad():
                _, _, next_val = policy(history_states)
                next_val = next_val[:, -1, :].item()
                
            returns = compute_gae(rewards, values, next_val, GAMMA, LAMBDA)
            advantages = [r - v for r, v in zip(returns, values)]
            
            all_states.append(states)
            all_actions.append(actions)
            all_logprobs.append(logprobs)
            all_returns.append(returns)
            all_advantages.append(advantages)
            
        print(f"Epoch {epoch:3d} | Avg Reward: {batch_reward/EPISODES_PER_BATCH:7.1f}")
        
        # 2. PPO Update (sequence level)
        policy.train()
        
        # Convert to tensors (B, T, D)
        states_t = torch.tensor(np.array(all_states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(np.array(all_actions), dtype=torch.float32, device=device)
        logprobs_t = torch.tensor(np.array(all_logprobs), dtype=torch.float32, device=device)
        returns_t = torch.tensor(np.array(all_returns), dtype=torch.float32, device=device).unsqueeze(-1)
        advantages_t = torch.tensor(np.array(all_advantages), dtype=torch.float32, device=device).unsqueeze(-1)
        
        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        for _ in range(K_EPOCHS):
            # Forward pass over full sequences
            mean, log_std, values = policy(states_t)
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            
            new_logprobs = dist.log_prob(actions_t).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().mean()
            
            ratios = torch.exp(new_logprobs - logprobs_t.unsqueeze(-1))
            
            surr1 = ratios * advantages_t
            surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages_t
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, returns_t)
            
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    os.makedirs("../checkpoints", exist_ok=True)
    torch.save(policy.state_dict(), "../checkpoints/bdh_ppo_policy.pt")
    print("PPO training complete. Saved to '../checkpoints/bdh_ppo_policy.pt'")
    env.close()

if __name__ == "__main__":
    train_ppo()
