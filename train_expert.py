import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 100000
MAX_EPISODES = 200
HORIZON = 200

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state, dtype=np.float32),
                np.array(action, dtype=np.float32),
                np.array(reward, dtype=np.float32),
                np.array(next_state, dtype=np.float32),
                np.array(done, dtype=np.float32))
                
    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=2.0):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action
        
    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
    def forward(self, state, action):
        q = torch.relu(self.l1(torch.cat([state, action], dim=-1)))
        q = torch.relu(self.l2(q))
        return self.l3(q)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# Main DDPG Training
def train_ddpg():
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    actor = Actor(state_dim, action_dim).to(device)
    actor_target = Actor(state_dim, action_dim).to(device)
    actor_target.load_state_dict(actor.state_dict())
    actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    
    critic = Critic(state_dim, action_dim).to(device)
    critic_target = Critic(state_dim, action_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())
    critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)
    
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    
    print("Training DDPG expert on Pendulum-v1 (Regime A)...")
    
    for episode in range(MAX_EPISODES):
        state, info = env.reset()
        episode_reward = 0
        
        for t in range(HORIZON):
            # Select action with exploration noise
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            actor.eval()
            with torch.no_grad():
                action = actor(state_tensor).cpu().numpy()[0]
            actor.train()
            
            # Add noise for exploration
            action = action + np.random.normal(0, 0.2, size=action_dim)
            action = np.clip(action, -2.0, 2.0)
            
            # Step env
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            # Update network weights
            if len(replay_buffer) > BATCH_SIZE:
                s, a, r, s_next, d = replay_buffer.sample(BATCH_SIZE)
                
                s_t = torch.tensor(s, device=device)
                a_t = torch.tensor(a, device=device)
                r_t = torch.tensor(r, device=device).unsqueeze(1)
                sn_t = torch.tensor(s_next, device=device)
                d_t = torch.tensor(d, device=device).unsqueeze(1)
                
                # 1. Update Critic
                with torch.no_grad():
                    target_q = r_t + (1 - d_t) * GAMMA * critic_target(sn_t, actor_target(sn_t))
                current_q = critic(s_t, a_t)
                critic_loss = nn.functional.mse_loss(current_q, target_q)
                
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                
                # 2. Update Actor
                actor_loss = -critic(s_t, actor(s_t)).mean()
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
                # 3. Soft updates of target networks
                soft_update(actor_target, actor, TAU)
                soft_update(critic_target, critic, TAU)
                
            if done:
                break
                
        if episode % 20 == 0 or episode == MAX_EPISODES - 1:
            print(f"Episode {episode:3d} | Reward: {episode_reward:7.1f}")
            
        # Stop early if solved (Pendulum is considered solved around -200 reward)
        if episode_reward > -180:
            print(f"Solved at episode {episode} with reward {episode_reward:.1f}!")
            break
            
    # Save the trained actor network (policy)
    torch.save(actor.state_dict(), "expert_policy.pt")
    print("Expert policy saved to 'expert_policy.pt'")
    env.close()

if __name__ == "__main__":
    train_ddpg()
