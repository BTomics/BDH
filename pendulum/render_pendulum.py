import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import bdh
from env import PendulumRegimeEnv
from policy_models import BDHPolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HORIZON = 200
SWITCH_TIMESTEP = 100

def render_rollout(policy_path, config, output_file="../figs/pendulum_rollout.gif"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Initialize env with render_mode="rgb_array"
    env = PendulumRegimeEnv(render_mode="rgb_array")
    env.set_regime("A")
    obs, info = env.reset(seed=42)
    
    # Load policy
    policy = BDHPolicy(config).to(device)
    try:
        policy.load_state_dict(torch.load(policy_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Could not find {policy_path}. Did you move it to checkpoints/?")
        return
    policy.eval()
    
    history_states = torch.tensor([obs], dtype=torch.float32, device=device).unsqueeze(0)
    
    frames = []
    frames.append(env.env.render())
    
    print("Running rollout for rendering...")
    for t in range(HORIZON):
        if t == SWITCH_TIMESTEP:
            env.set_regime("B") # Switch gravity
            
        with torch.no_grad():
            preds, _ = policy(history_states)
            action = preds[:, -1, :].cpu().numpy()[0]
            
        action = np.clip(action, -2.0, 2.0)
        obs_next, reward, terminated, truncated, info = env.step(action)
        
        frames.append(env.env.render())
        
        obs_tensor = torch.tensor([obs_next], dtype=torch.float32, device=device).unsqueeze(0)
        history_states = torch.cat([history_states, obs_tensor], dim=1)
        
    env.close()
    
    print(f"Saving {len(frames)} frames to {output_file}...")
    fig = plt.figure()
    plt.axis('off')
    im = plt.imshow(frames[0])
    
    def update(frame):
        im.set_array(frame)
        return [im]
        
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    ani.save(output_file, writer='pillow', fps=20)
    print(f"Video saved to {output_file}")

if __name__ == "__main__":
    policy_path = "../checkpoints/bdh_policy.pt"
    bdh_config = bdh.BDHConfig(
        n_layer=3, n_embd=64, n_head=4, mlp_internal_dim_multiplier=8,
        state_dim=3, action_dim=1
    )
    render_rollout(policy_path, bdh_config)
