import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import torch
import numpy as np
import matplotlib.pyplot as plt
import bdh
from env import LunarLanderRegimeEnv
from train_expert_lander import Actor as ExpertActor
from policy_models import BDHPolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HORIZON = 500
SWITCH_TIMESTEP = 60
REGIME_B = "B"

def run_closed_loop_rollout(policy_model, model_name):
    env = LunarLanderRegimeEnv(privileged=False)
    env.set_regime("A")
    obs, info = env.reset(seed=42)
    
    states = [obs]
    actions = []
    rewards = []
    
    history_states = torch.tensor([obs], dtype=torch.float32, device=device).unsqueeze(0)
    
    for t in range(HORIZON):
        if t == SWITCH_TIMESTEP:
            env.set_regime(REGIME_B)
            
        if model_name == "Expert":
            health = env.regimes[env.current_regime]["health"]
            priv_obs = np.append(obs, health).astype(np.float32)
            state_tensor = torch.tensor(priv_obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = policy_model(state_tensor).cpu().numpy()[0]
        else:
            with torch.no_grad():
                preds, _ = policy_model(history_states)
                action = preds[:, -1, :].cpu().numpy()[0]
                
        action = np.clip(action, -1.0, 1.0)
        obs_next, reward, terminated, truncated, info = env.step(action)
        
        states.append(obs_next)
        actions.append(action)
        rewards.append(reward)
        
        obs = obs_next
        obs_tensor = torch.tensor([obs], dtype=torch.float32, device=device).unsqueeze(0)
        history_states = torch.cat([history_states, obs_tensor], dim=1)
        
        if terminated or truncated:
            # Pad rewards with 0s to reach HORIZON so we can plot them side-by-side
            pad_len = HORIZON - len(rewards)
            if pad_len > 0:
                rewards.extend([0.0] * pad_len)
            break
            
    # If the episode naturally didn't end (e.g., hovering), rewards list is already HORIZON length
    env.close()
    return np.array(rewards)

if __name__ == "__main__":
    expert = ExpertActor(state_dim=9, action_dim=2).to(device)
    if os.path.exists("checkpoints/expert_lander.pt"):
        expert.load_state_dict(torch.load("checkpoints/expert_lander.pt", map_location=device))
    expert.eval()
    
    bdh_config = bdh.BDHConfig(
        n_layer=3, n_embd=128, n_head=4, mlp_internal_dim_multiplier=4,
        state_dim=8, action_dim=2
    )
    
    bdh_dagger = BDHPolicy(bdh_config).to(device)
    if os.path.exists("checkpoints/bdh_dagger_lander.pt"):
        bdh_dagger.load_state_dict(torch.load("checkpoints/bdh_dagger_lander.pt", map_location=device))
    bdh_dagger.eval()
    
    results = {}
    
    print("\nRunning closed-loop evaluations for LunarLander...")
    if os.path.exists("checkpoints/expert_lander.pt"): results["Expert (Privileged)"] = run_closed_loop_rollout(expert, "Expert")
    if os.path.exists("checkpoints/bdh_dagger_lander.pt"): results["BDH (DAgger)"] = run_closed_loop_rollout(bdh_dagger, "BDH")
    
    print("\n=== CLOSED-LOOP EPISODE RETURNS ===")
    for name, rewards in results.items():
        print(f"{name:<20} : Return = {np.sum(rewards):.1f}")
    print("===================================\n")
    
    os.makedirs("figs", exist_ok=True)
    plt.figure(figsize=(12, 6))
    
    colors = {
        "Expert (Privileged)": "black",
        "BDH (DAgger)": "royalblue",
    }
    
    styles = {
        "Expert (Privileged)": ":",
        "BDH (DAgger)": "-",
    }
    
    for name, rewards in results.items():
        # Compute cumulative reward over time for a smoother plot since step rewards in LunarLander are highly variable
        cumulative_rewards = np.cumsum(rewards)
        plt.plot(cumulative_rewards, label=name, color=colors.get(name, "gray"), linestyle=styles.get(name, "-"), linewidth=2)
        
    plt.axvline(x=SWITCH_TIMESTEP, color="red", linestyle=":", label="Thruster Failure!")
    plt.title("Cumulative Reward Recovery Under Thruster Failure")
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("figs/lander_recovery.png", dpi=300)
    plt.close()
    print("Recovery plot saved to 'figs/lander_recovery.png'")
