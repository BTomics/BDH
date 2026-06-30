import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import bdh
from env import PendulumRegimeEnv
from train_expert import Actor
from policy_models import BDHPolicy, TransformerPolicy, GRUPolicy, BDHActorCritic

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
        elif "PPO" in model_name:
            with torch.no_grad():
                action_mean, _, _ = policy_model(history_states)
                action = action_mean[:, -1, :].cpu().numpy()[0]
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
    files = ["checkpoints/expert_policy.pt", "checkpoints/bdh_policy.pt", "checkpoints/transformer_policy.pt", "checkpoints/gru_policy.pt", "checkpoints/bdh_dagger_policy.pt", "checkpoints/bdh_ppo_policy.pt"]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        print(f"Warning: Missing checkpoints: {missing}.")
        
    print("Loading cloned policy checkpoints...")
    
    # 1. Expert Policy
    expert = Actor(state_dim=3, action_dim=1).to(device)
    if os.path.exists("checkpoints/expert_policy.pt"):
        expert.load_state_dict(torch.load("checkpoints/expert_policy.pt", map_location=device))
    expert.eval()
    
    # BDH Config
    bdh_config = bdh.BDHConfig(
        n_layer=3, n_embd=64, n_head=4, mlp_internal_dim_multiplier=8,
        state_dim=3, action_dim=1
    )
    
    # 2. BDH Policy (BC)
    bdh_policy = BDHPolicy(bdh_config).to(device)
    if os.path.exists("checkpoints/bdh_policy.pt"):
        bdh_policy.load_state_dict(torch.load("checkpoints/bdh_policy.pt", map_location=device))
    bdh_policy.eval()
    
    # 3. Transformer Policy
    tf_policy = TransformerPolicy(state_dim=3, action_dim=1, n_embd=96, n_head=4, n_layer=3).to(device)
    if os.path.exists("checkpoints/transformer_policy.pt"):
        tf_policy.load_state_dict(torch.load("checkpoints/transformer_policy.pt", map_location=device))
    tf_policy.eval()
    
    # 4. GRU Policy
    gru_policy = GRUPolicy(state_dim=3, action_dim=1, n_embd=128, n_layer=2).to(device)
    if os.path.exists("checkpoints/gru_policy.pt"):
        gru_policy.load_state_dict(torch.load("checkpoints/gru_policy.pt", map_location=device))
    gru_policy.eval()
    
    # 5. BDH DAgger Policy
    bdh_dagger = BDHPolicy(bdh_config).to(device)
    if os.path.exists("checkpoints/bdh_dagger_policy.pt"):
        bdh_dagger.load_state_dict(torch.load("checkpoints/bdh_dagger_policy.pt", map_location=device))
    bdh_dagger.eval()
    
    # 6. BDH PPO Policy
    bdh_ppo = BDHActorCritic(bdh_config).to(device)
    if os.path.exists("checkpoints/bdh_ppo_policy.pt"):
        bdh_ppo.load_state_dict(torch.load("checkpoints/bdh_ppo_policy.pt", map_location=device))
    bdh_ppo.eval()
    
    # Run closed loop evaluations
    results = {}
    
    print("\nRunning closed-loop evaluations...")
    if os.path.exists("checkpoints/expert_policy.pt"): results["Expert"] = run_closed_loop_rollout(expert, "Expert")
    if os.path.exists("checkpoints/bdh_policy.pt"): results["BDH (BC)"] = run_closed_loop_rollout(bdh_policy, "BDH (BC)", freeze_timestep=None)
    if os.path.exists("checkpoints/transformer_policy.pt"): results["Transformer (BC)"] = run_closed_loop_rollout(tf_policy, "Transformer")
    if os.path.exists("checkpoints/gru_policy.pt"): results["GRU (BC)"] = run_closed_loop_rollout(gru_policy, "GRU")
    if os.path.exists("checkpoints/bdh_dagger_policy.pt"): results["BDH (DAgger)"] = run_closed_loop_rollout(bdh_dagger, "BDH (DAgger)", freeze_timestep=None)
    if os.path.exists("checkpoints/bdh_ppo_policy.pt"): results["BDH (PPO)"] = run_closed_loop_rollout(bdh_ppo, "BDH (PPO)")
    
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
        "BDH (BC)": "red",
        "Transformer (BC)": "forestgreen",
        "GRU (BC)": "darkorange",
        "BDH (DAgger)": "royalblue",
        "BDH (PPO)": "purple"
    }
    
    styles = {
        "Expert": ":",
        "BDH (BC)": "-",
        "Transformer (BC)": "-",
        "GRU (BC)": "-",
        "BDH (DAgger)": "-",
        "BDH (PPO)": "-"
    }
    
    for name, (rewards, states) in results.items():
        plt.plot(rewards, label=name, color=colors.get(name, "gray"), linestyle=styles.get(name, "-"), linewidth=2)
        
    plt.axvline(x=SWITCH_TIMESTEP, color="red", linestyle=":", label="Dynamics Switch (A -> B)")
    plt.title("Closed-Loop Step Reward Recovery Under Dynamics Switch")
    plt.xlabel("Timestep")
    plt.ylabel("Step Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("figs/closed_loop_recovery.png", dpi=300)
    plt.close()
    print("Closed-loop recovery plot saved to 'figs/closed_loop_recovery.png'")
