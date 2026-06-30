import sys
import os
import time
import torch
import numpy as np
import gymnasium as gym

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import bdh
from policy_models import BDHPolicy
from env import PendulumRegimeEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HORIZON = 200
SWITCH_TIMESTEP = 100

def run_live(policy_path, config):
    # Initialize env with render_mode="human" to open a live window
    env = PendulumRegimeEnv(render_mode="human")
    env.set_regime("A")
    obs, info = env.reset(seed=42)
    
    # Load policy
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
    print("Regime A: Normal Gravity (g=10)")
    
    for t in range(HORIZON):
        if t == SWITCH_TIMESTEP:
            print("\n*** DYNAMICS SWITCH ***")
            print("Regime B: Increased Gravity (g=13)")
            env.set_regime("B")
            
        with torch.no_grad():
            preds, _ = policy(history_states)
            action = preds[:, -1, :].cpu().numpy()[0]
            
        action = np.clip(action, -2.0, 2.0)
        obs_next, reward, terminated, truncated, info = env.step(action)
        
        obs_tensor = torch.tensor([obs_next], dtype=torch.float32, device=device).unsqueeze(0)
        history_states = torch.cat([history_states, obs_tensor], dim=1)
        
        # Add a small delay so the animation runs at a viewable speed (~30 fps)
        time.sleep(0.03)
        
    print("\nRollout finished. Closing window...")
    env.close()

if __name__ == "__main__":
    bdh_config = bdh.BDHConfig(
        n_layer=3, n_embd=64, n_head=4, mlp_internal_dim_multiplier=8,
        state_dim=3, action_dim=1
    )
    
    policy_path = "../checkpoints/bdh_dagger_policy.pt"
    run_live(policy_path, bdh_config)
