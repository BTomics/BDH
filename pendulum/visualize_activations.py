import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import bdh
from env import PendulumRegimeEnv
from policy_models import BDHPolicy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HORIZON = 200
SWITCH_TIMESTEP = 100

def generate_activations(policy_path, config):
    # Initialize env
    env = PendulumRegimeEnv()
    env.set_regime("A")
    obs, info = env.reset(seed=42)
    
    # Load policy
    policy = BDHPolicy(config).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()
    
    history_states = torch.tensor([obs], dtype=torch.float32, device=device).unsqueeze(0)
    
    print("Running rollout to collect activations...")
    
    rewards = []
    cos_thetas = []
    
    for t in range(HORIZON):
        if t == SWITCH_TIMESTEP:
            env.set_regime("B") # Switch gravity
            
        with torch.no_grad():
            preds, _ = policy(history_states)
            action = preds[:, -1, :].cpu().numpy()[0]
            
        action = np.clip(action, -2.0, 2.0)
        obs_next, reward, terminated, truncated, info = env.step(action)
        
        rewards.append(reward)
        cos_thetas.append(obs[0]) # obs[0] is cos(theta)
        
        obs_tensor = torch.tensor([obs_next], dtype=torch.float32, device=device).unsqueeze(0)
        history_states = torch.cat([history_states, obs_tensor], dim=1)
        obs = obs_next
        
    env.close()
    
    print("Extracting node activations from all layers...")
    with torch.no_grad():
        _, all_activations = policy.forward_with_activations(history_states[:, :-1, :])
        
    return all_activations, np.array(rewards), np.array(cos_thetas)

def plot_enhanced_activations(all_activations, rewards, cos_thetas, n_layer=3, n_head=0, filename="../figs/enhanced_activations.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Create subplots: Top for Rewards/Angle, Bottom ones for Layers
    fig, axes = plt.subplots(n_layer + 1, 1, figsize=(14, 3 + 4 * n_layer), sharex=True, gridspec_kw={'height_ratios': [1.5] + [2]*n_layer})
    
    # --- Top Plot: Reward and Cos(Theta) ---
    ax_perf = axes[0]
    ax_perf.plot(rewards, color='red', label='Step Reward', linewidth=2)
    ax_perf.set_ylabel("Reward", color='red')
    ax_perf.tick_params(axis='y', labelcolor='red')
    
    ax_angle = ax_perf.twinx()
    ax_angle.plot(cos_thetas, color='blue', label='cos(theta)', alpha=0.5, linestyle='--')
    ax_angle.set_ylabel("cos(theta) [1.0 is upright]", color='blue')
    ax_angle.tick_params(axis='y', labelcolor='blue')
    
    ax_perf.axvline(x=SWITCH_TIMESTEP, color="black", linestyle="--", linewidth=2, label="Gravity Doubles!")
    ax_perf.set_title(f"BDH (DAgger) Performance vs Internal Node Activations", fontsize=14, pad=15)
    
    # Combine legends
    lines_1, labels_1 = ax_perf.get_legend_handles_labels()
    lines_2, labels_2 = ax_angle.get_legend_handles_labels()
    ax_perf.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower left')
    
    # --- Bottom Plots: Activations ---
    for layer in range(n_layer):
        act_matrix = all_activations[layer][0, n_head].cpu().numpy()
        
        # 1. Smooth over time axis (T) to reduce noise
        act_matrix = gaussian_filter1d(act_matrix, sigma=1.5, axis=0)
        
        # 2. Sort nodes by variance (highest variance at the top)
        variances = np.var(act_matrix, axis=0)
        sorted_indices = np.argsort(variances)[::-1]
        act_matrix = act_matrix[:, sorted_indices]
        
        # 3. Transpose to (N, T) for imshow
        act_matrix = act_matrix.T
        
        ax = axes[layer + 1]
        im = ax.imshow(act_matrix, cmap="magma", aspect="auto", interpolation="nearest")
        
        ax.axvline(x=SWITCH_TIMESTEP, color="cyan", linestyle="--", linewidth=1.5)
        ax.set_title(f"Layer {layer + 1} Sorted Node Activations (Head {n_head})", fontsize=12)
        ax.set_ylabel("Node Index (Sorted by Variance)")
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Smoothed Activation")
        
    axes[-1].set_xlabel("Timestep", fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Enhanced visualization saved to '{filename}'")

if __name__ == "__main__":
    policy_path = "../checkpoints/bdh_dagger_policy.pt"
    if not os.path.exists(policy_path):
        print(f"Error: Policy checkpoint '{policy_path}' not found. Please run train_dagger.py first.")
        exit(1)
        
    bdh_config = bdh.BDHConfig(
        n_layer=3, n_embd=64, n_head=4, mlp_internal_dim_multiplier=8,
        state_dim=3, action_dim=1
    )
    
    activations, rewards, cos_thetas = generate_activations(policy_path, bdh_config)
    plot_enhanced_activations(activations, rewards, cos_thetas, n_layer=bdh_config.n_layer, n_head=0)
