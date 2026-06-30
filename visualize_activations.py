import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import bdh
from env_wrapper import PendulumRegimeEnv
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
    
    # Track states
    history_states = torch.tensor([obs], dtype=torch.float32, device=device).unsqueeze(0) # (1, 1, 3)
    
    print("Running rollout to collect activations...")
    
    # Run the rollout
    for t in range(HORIZON):
        if t == SWITCH_TIMESTEP:
            env.set_regime("B") # Switch gravity
            
        with torch.no_grad():
            preds, _ = policy(history_states)
            action = preds[:, -1, :].cpu().numpy()[0]
            
        action = np.clip(action, -2.0, 2.0)
        obs_next, reward, terminated, truncated, info = env.step(action)
        
        # Update history
        obs_tensor = torch.tensor([obs_next], dtype=torch.float32, device=device).unsqueeze(0)
        history_states = torch.cat([history_states, obs_tensor], dim=1)
        
    env.close()
    
    # Run a final forward pass with activations over the complete sequence
    print("Extracting node activations from all layers...")
    with torch.no_grad():
        _, all_activations = policy.forward_with_activations(history_states[:, :-1, :])
        
    # all_activations is a list of len (n_layer) containing tensors of shape (1, nh, T, N)
    return all_activations

def plot_layer_activations(all_activations, n_layer=3, n_head=0, filename="figs/node_activations.png"):
    """
    Plots the node activations for all layers (for a specific head) over time.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    fig, axes = plt.subplots(n_layer, 1, figsize=(12, 4 * n_layer), sharex=True)
    if n_layer == 1:
        axes = [axes]
        
    for layer in range(n_layer):
        # Shape of activation at layer: (1, nh, T, N)
        # Extract first batch, specified head -> (T, N)
        act_matrix = all_activations[layer][0, n_head].cpu().numpy()
        
        # Transpose to (N, T) for plotting (nodes on y-axis, time on x-axis)
        act_matrix = act_matrix.T 
        
        ax = axes[layer]
        im = ax.imshow(act_matrix, cmap="magma", aspect="auto", interpolation="nearest")
        
        # Mark the dynamics switch point
        ax.axvline(x=SWITCH_TIMESTEP, color="cyan", linestyle="--", linewidth=1.5, label="Dynamics Switch")
        
        ax.set_title(f"Layer {layer + 1} Sparse Node Activations (Head {n_head})")
        ax.set_ylabel("Node Index (0-N)")
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label="Activation Intensity")
        
    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Activation visualization saved to '{filename}'")

if __name__ == "__main__":
    policy_path = "bdh_policy.pt"
    if not os.path.exists(policy_path):
        print(f"Error: Policy checkpoint '{policy_path}' not found. Please run clone_policy.py first.")
        exit(1)
        
    bdh_config = bdh.BDHConfig(
        n_layer=3, n_embd=64, n_head=4, mlp_internal_dim_multiplier=8,
        state_dim=3, action_dim=1
    )
    
    # 1. Run rollout and get activations
    activations = generate_activations(policy_path, bdh_config)
    
    # 2. Plot activations
    plot_layer_activations(activations, n_layer=bdh_config.n_layer, n_head=0)
