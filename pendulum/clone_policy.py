import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import os
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import bdh
from train_expert import Actor
from policy_models import BDHPolicy, TransformerPolicy, GRUPolicy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1337)

# Hyperparameters
NUM_BC_EPISODES = 200
VAL_BC_EPISODES = 40
HORIZON = 200
EPOCHS = 1000
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

# 1. Generate Expert Trajectories for BC
def collect_expert_trajectories(expert_path, num_episodes):
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Load expert actor
    expert = Actor(state_dim, action_dim).to(device)
    expert.load_state_dict(torch.load(expert_path, map_location=device))
    expert.eval()
    
    all_states = []
    all_actions = []
    
    print(f"Collecting {num_episodes} episodes of expert rollouts for Behavior Cloning...")
    
    for ep in range(num_episodes):
        state, info = env.reset()
        states = []
        actions = []
        
        for t in range(HORIZON):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = expert(state_tensor).cpu().numpy()[0]
                
            states.append(state)
            actions.append(action)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            
            if terminated or truncated:
                break
                
        all_states.append(states)
        all_actions.append(actions)
        
    env.close()
    return np.array(all_states, dtype=np.float32), np.array(all_actions, dtype=np.float32)

def train_policy(model, name, train_s, train_a, val_s, val_a):
    print(f"\nBehavior Cloning into {name}...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    best_val_loss = float("inf")
    best_state = None
    
    for epoch in range(EPOCHS):
        model.train()
        # Sample batch
        ix = np.random.randint(0, train_s.shape[0], BATCH_SIZE)
        s_batch = torch.tensor(train_s[ix], device=device)
        a_batch = torch.tensor(train_a[ix], device=device)
        
        preds, loss = model(s_batch, a_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                val_ix = np.random.randint(0, val_s.shape[0], BATCH_SIZE)
                s_val = torch.tensor(val_s[val_ix], device=device)
                a_val = torch.tensor(val_a[val_ix], device=device)
                _, val_loss = model(s_val, a_val)
                
            print(f"Epoch {epoch:4d} | Train MSE: {loss.item():.6f} | Val MSE: {val_loss.item():.6f}")
            
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save(model.state_dict(), f"{name.lower().replace(' ', '_')}_policy.pt")
    print(f"Saved {name} policy checkpoint to '{name.lower().replace(' ', '_')}_policy.pt'")
    return model

if __name__ == "__main__":
    expert_path = "../checkpoints/expert_policy.pt"
    if not os.path.exists(expert_path):
        print(f"Error: Expert policy '{expert_path}' not found. Please train the expert first.")
        exit(1)
        
    train_s, train_a = collect_expert_trajectories(expert_path, NUM_BC_EPISODES)
    val_s, val_a = collect_expert_trajectories(expert_path, VAL_BC_EPISODES)
    
    # Initialize networks
    bdh_config = bdh.BDHConfig(
        n_layer=3, n_embd=64, n_head=4, mlp_internal_dim_multiplier=8,
        state_dim=3, action_dim=1
    )
    
    bdh_policy = BDHPolicy(bdh_config).to(device)
    tf_policy = TransformerPolicy(state_dim=3, action_dim=1, n_embd=96, n_head=4, n_layer=3).to(device)
    gru_policy = GRUPolicy(state_dim=3, action_dim=1, n_embd=128, n_layer=2).to(device)
    
    # Train
    train_policy(bdh_policy, "BDH", train_s, train_a, val_s, val_a)
    train_policy(tf_policy, "Transformer", train_s, train_a, val_s, val_a)
    train_policy(gru_policy, "GRU", train_s, train_a, val_s, val_a)
