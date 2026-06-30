import sys
import os
import time
import torch
import numpy as np
import gymnasium as gym

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import bdh
from policy_models import BDHPolicy
from env import LunarLanderRegimeEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HORIZON = 500
SWITCH_TIMESTEP = 150 # Induce failure at step 150

def run_live(policy_path, config):
    env = LunarLanderRegimeEnv(render_mode="human", privileged=False)
    env.set_regime("A")
    obs, info = env.reset(seed=42)
    
    policy = BDHPolicy(config).to(device)
    try:
        policy.load_state_dict(torch.load(policy_path, map_location=device))
        print(f"Successfully loaded policy from {policy_path}")
    except FileNotFoundError:
        print(f"Error: Could not find {policy_path}.")
        return
    policy.eval()
    
    history_states = torch.tensor([obs], dtype=torch.float32, device=device).unsqueeze(0)
    
    print("\nStarting live rendering... (Window should open!)")
    print("Regime A: Normal Flight (Both thrusters 100%)")
    
    total_reward = 0
    
    for t in range(HORIZON):
        if t == SWITCH_TIMESTEP:
            print("\n*** CRITICAL ALERT: THRUSTER FAILURE ***")
            print("Regime B: Left side-thruster power dropped to 20%!")
            env.set_regime("B")
            
        with torch.no_grad():
            preds, _ = policy(history_states)
            action = preds[:, -1, :].cpu().numpy()[0]
            
        action = np.clip(action, -1.0, 1.0)
        obs_next, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        obs_tensor = torch.tensor([obs_next], dtype=torch.float32, device=device).unsqueeze(0)
        history_states = torch.cat([history_states, obs_tensor], dim=1)
        
        time.sleep(0.02)
        
        if terminated or truncated:
            print(f"Episode finished! Total Reward: {total_reward:.1f}")
            break
            
    print("\nClosing window...")
    env.close()

if __name__ == "__main__":
    bdh_config = bdh.BDHConfig(
        n_layer=3, n_embd=128, n_head=4, mlp_internal_dim_multiplier=4,
        state_dim=8, action_dim=2
    )
    
    policy_path = "checkpoints/bdh_dagger_lander.pt"
    run_live(policy_path, bdh_config)
