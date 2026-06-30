import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import bdh
from env_wrapper import PendulumRegimeEnv
from train_expert import Actor
from policy_models import BDHPolicy, TransformerPolicy, GRUPolicy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HORIZON = 200
SWITCH_TIMESTEP = 100
REGIME_B = "B" # Double gravity

def run_closed_loop_rollout(policy_model, model_name, freeze_timestep=None):
    """
    Runs a closed-loop rollout. The policy predicts actions from history_states.
    Dynamics switch A -> B at SWITCH_TIMESTEP.
    """
    env = PendulumRegimeEnv()
    env.set_regime("A")
    obs, info = env.reset(seed=42)
    
    states = [obs]
    actions = []
    rewards = []
    
    # Track state history for sequence policies
    history_states = torch.tensor([obs], dtype=torch.float32, device=device).unsqueeze(0) # (1, 1, 3)
    
    for t in range(HORIZON):
        if t == SWITCH_TIMESTEP:
            env.set_regime(REGIME_B)
            
        # Get action from policy
        if model_name == "Expert":
            state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = policy_model(state_tensor).cpu().numpy()[0]
        else:
            with torch.no_grad():
                if "BDH" in model_name:
                    preds, _ = policy_model(history_states, freeze_timestep=freeze_timestep)
                else:
                    preds, _ = policy_model(history_states)
                action = preds[:, -1, :].cpu().numpy()[0]
                
        # Clip action to gym limits
        action = np.clip(action, -2.0, 2.0)
        
        # Step env
        obs_next, reward, terminated, truncated, info = env.step(action)
        
        # Record
        states.append(obs_next)
        actions.append(action)
        rewards.append(reward)
        
        # Update history
        obs = obs_next
        obs_tensor = torch.tensor([obs], dtype=torch.float32, device=device).unsqueeze(0)
        history_states = torch.cat([history_states, obs_tensor], dim=1)
        
    env.close()
    return np.array(rewards), np.array(states)

if __name__ == "__main__":
    # Check checkpoints
    files = ["expert_policy.pt", "bdh_policy.pt", "transformer_policy.pt", "gru_policy.pt"]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        print(f"Error: Missing checkpoints: {missing}. Please run train_expert.py and clone_policy.py first.")
        exit(1)
        
    print("Loading cloned policy checkpoints...")
    
    # 1. Expert Policy
    expert = Actor(state_dim=3, action_dim=1).to(device)
    expert.load_state_dict(torch.load("expert_policy.pt", map_location=device))
    expert.eval()
    
    # 2. BDH Policy
    bdh_config = bdh.BDHConfig(
        n_layer=3, n_embd=64, n_head=4, mlp_internal_dim_multiplier=8,
        state_dim=3, action_dim=1
    )
    bdh_policy = BDHPolicy(bdh_config).to(device)
    bdh_policy.load_state_dict(torch.load("bdh_policy.pt", map_location=device))
    bdh_policy.eval()
    
    # 3. Transformer Policy
    tf_policy = TransformerPolicy(state_dim=3, action_dim=1, n_embd=96, n_head=4, n_layer=3).to(device)
    tf_policy.load_state_dict(torch.load("transformer_policy.pt", map_location=device))
    tf_policy.eval()
    
    # 4. GRU Policy
    gru_policy = GRUPolicy(state_dim=3, action_dim=1, n_embd=128, n_layer=2).to(device)
    gru_policy.load_state_dict(torch.load("gru_policy.pt", map_location=device))
    gru_policy.eval()
    
    # Run closed loop evaluations
    results = {}
    
    print("\nRunning closed-loop evaluations...")
    results["Expert"] = run_closed_loop_rollout(expert, "Expert")
    results["BDH (Live-sigma)"] = run_closed_loop_rollout(bdh_policy, "BDH (Live-sigma)", freeze_timestep=None)
    results["BDH (Frozen-sigma)"] = run_closed_loop_rollout(bdh_policy, "BDH (Frozen-sigma)", freeze_timestep=SWITCH_TIMESTEP)
    results["Transformer"] = run_closed_loop_rollout(tf_policy, "Transformer")
    results["GRU"] = run_closed_loop_rollout(gru_policy, "GRU")
    
    # Print episode returns
    print("\n=== CLOSED-LOOP EPISODE RETURNS ===")
    for name, (rewards, states) in results.items():
        print(f"{name:<18} : Return = {np.sum(rewards):.1f}")
    print("===================================\n")
    
    # Plot Step Rewards
    os.makedirs("figs", exist_ok=True)
    plt.figure(figsize=(12, 6))
    
    colors = {
        "Expert": "black",
        "BDH (Live-sigma)": "royalblue",
        "BDH (Frozen-sigma)": "dodgerblue",
        "Transformer": "forestgreen",
        "GRU": "darkorange"
    }
    
    styles = {
        "Expert": ":",
        "BDH (Live-sigma)": "-",
        "BDH (Frozen-sigma)": "--",
        "Transformer": "-",
        "GRU": "-"
    }
    
    for name, (rewards, states) in results.items():
        plt.plot(rewards, label=name, color=colors[name], linestyle=styles[name], linewidth=2)
        
    plt.axvline(x=SWITCH_TIMESTEP, color="red", linestyle=":", label="Dynamics Switch (A -> B)")
    plt.title("Closed-Loop Step Reward Recovery Under Dynamics Switch")
    plt.xlabel("Timestep")
    plt.ylabel("Step Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("figs/closed_loop_recovery.png", dpi=300)
    plt.close()
    print("Closed-loop recovery plot saved to 'figs/closed_loop_recovery.png'")
