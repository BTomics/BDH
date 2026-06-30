import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import random
import torch
import numpy as np
import bdh
from train_expert_lander import Actor as ExpertActor
from policy_models import BDHPolicy
from env import LunarLanderRegimeEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

DAGGER_ITERS = 10
EPISODES_PER_ITER = 30
HORIZON = 500
EPOCHS_PER_ITER = 300
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

def get_expert():
    expert_path = "checkpoints/expert_lander.pt"
    if not os.path.exists(expert_path):
        raise FileNotFoundError("Run train_expert_lander.py first to generate the privileged expert.")
    
    expert = ExpertActor(9, 2, max_action=1.0).to(device)
    expert.load_state_dict(torch.load(expert_path, map_location=device))
    expert.eval()
    return expert

def run_dagger():
    expert = get_expert()
    
    # Unprivileged environment for the BDH policy to act in.
    env = LunarLanderRegimeEnv(privileged=False)
    
    bdh_config = bdh.BDHConfig(
        n_layer=3, n_embd=128, n_head=4, mlp_internal_dim_multiplier=4,
        state_dim=8, action_dim=2
    )
    policy = BDHPolicy(bdh_config).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    dataset_s = []
    dataset_a = []
    
    print("Starting DAgger Training Loop for LunarLander...")
    
    for i in range(DAGGER_ITERS):
        print(f"\n--- DAgger Iteration {i+1}/{DAGGER_ITERS} ---")
        
        print(f"Collecting {EPISODES_PER_ITER} episodes...")
        iter_states = []
        iter_expert_actions = []
        
        for ep in range(EPISODES_PER_ITER):
            # Randomize regime so BDH learns to infer both normal and failed thrusters
            regime = random.choice(["A", "B"])
            env.set_regime(regime)
            
            state, info = env.reset()
            health = env.regimes[regime]["health"]
            
            states = []
            expert_actions = []
            
            history_states = torch.tensor([state], dtype=torch.float32, device=device).unsqueeze(0)
            
            for t in range(HORIZON):
                # Query Expert (needs privileged state)
                priv_state = np.append(state, health).astype(np.float32)
                priv_state_tensor = torch.tensor(priv_state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    expert_a = expert(priv_state_tensor).cpu().numpy()[0]
                
                states.append(state)
                expert_actions.append(expert_a)
                
                # Determine next action
                if i == 0:
                    action = expert_a
                else:
                    policy.eval()
                    with torch.no_grad():
                        preds, _ = policy(history_states)
                        action = preds[:, -1, :].cpu().numpy()[0]
                
                action = np.clip(action, -1.0, 1.0)
                next_state, reward, terminated, truncated, info = env.step(action)
                
                state = next_state
                obs_tensor = torch.tensor([state], dtype=torch.float32, device=device).unsqueeze(0)
                history_states = torch.cat([history_states, obs_tensor], dim=1)
                
                if terminated or truncated:
                    break
                    
            # Padding short sequences to HORIZON so we can batch them easily
            # (Or we can just mask them, but padding by duplicating last state is simpler for this toy script)
            pad_len = HORIZON - len(states)
            if pad_len > 0:
                states.extend([states[-1]] * pad_len)
                expert_actions.extend([[0.0, 0.0]] * pad_len)
                
            iter_states.append(states)
            iter_expert_actions.append(expert_actions)
            
        dataset_s.extend(iter_states)
        dataset_a.extend(iter_expert_actions)
        
        train_s = np.array(dataset_s, dtype=np.float32)
        train_a = np.array(dataset_a, dtype=np.float32)
        print(f"Dataset Size: {train_s.shape[0]} trajectories")
        
        print(f"Training BDH Policy for {EPOCHS_PER_ITER} epochs...")
        policy.train()
        
        for epoch in range(EPOCHS_PER_ITER):
            ix = np.random.randint(0, train_s.shape[0], BATCH_SIZE)
            s_batch = torch.tensor(train_s[ix], device=device)
            a_batch = torch.tensor(train_a[ix], device=device)
            
            preds, loss = policy(s_batch, a_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0 or epoch == EPOCHS_PER_ITER - 1:
                print(f"  Epoch {epoch:4d} | Train MSE: {loss.item():.6f}")
                
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(policy.state_dict(), "checkpoints/bdh_dagger_lander.pt")
    print("DAgger training complete. Saved to 'checkpoints/bdh_dagger_lander.pt'")
    env.close()

if __name__ == "__main__":
    run_dagger()
