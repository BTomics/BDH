import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from env_wrapper import PendulumRegimeEnv
import bdh

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_switch_rollout(model, horizon: int = 200, switch_timestep: int = 100, regime_b: str = "B"):
    """
    Runs a single rollout of length 'horizon', switching from Regime A to 'regime_b' at 'switch_timestep'.
    No gradient updates are performed. We track the prediction error at each step.
    """
    model.eval()
    env = PendulumRegimeEnv()
    env.set_regime("A")
    
    obs, info = env.reset(seed=42)
    
    # Store trajectory data
    states = [obs]
    actions = []
    predictions = []
    errors = []
    
    # History for autoregressive input
    history_states = torch.tensor([obs], dtype=torch.float32, device=device).unsqueeze(0) # (1, 1, 3)
    history_actions = torch.zeros((1, 0, 1), dtype=torch.float32, device=device) # (1, 0, 1)
    
    print(f"Starting rollout. Will switch A -> {regime_b} at step {switch_timestep}...")
    
    for t in range(horizon):
        # Switch dynamics mid-rollout
        if t == switch_timestep:
            env.set_regime(regime_b)
            
        # Select action (sinusoidal force to ensure dynamic states)
        action_val = np.sin(t / 10.0) * 2.0
        action = np.array([action_val], dtype=np.float32)
        
        # Step the environment
        obs_next, reward, terminated, truncated, info = env.step(action)
        
        # Prepare inputs for the model (entire history up to t)
        action_tensor = torch.tensor([action], dtype=torch.float32, device=device).unsqueeze(0) # (1, 1, 1)
        history_actions = torch.cat([history_actions, action_tensor], dim=1) # (1, t+1, 1)
        
        # Model prediction for next state
        with torch.no_grad():
            preds, _ = model(history_states, history_actions)
            pred_next = preds[:, -1, :].cpu().numpy()[0] # Prediction for s_{t+1}
            
        # Calculate prediction error (MSE)
        error = np.mean((obs_next - pred_next) ** 2)
        
        # Record
        states.append(obs_next)
        actions.append(action)
        predictions.append(pred_next)
        errors.append(error)
        
        # Update history for next step
        obs_next_tensor = torch.tensor([obs_next], dtype=torch.float32, device=device).unsqueeze(0)
        history_states = torch.cat([history_states, obs_next_tensor], dim=1) # (1, t+2, 3)
        
    env.close()
    return np.array(errors), switch_timestep

def calculate_recovery_metrics(errors, switch_timestep, threshold_multiplier: float = 1.1):
    """
    PRO-85: Calculate recovery metrics.
    - Pre-switch baseline: average error in the 20 steps before the switch.
    - Time-to-recover: steps after the switch until error returns to within threshold_multiplier * baseline.
    - Area Under the Recovery Curve (AURC): sum of errors post-switch.
    """
    # 1. Pre-switch baseline error
    pre_switch_window = errors[max(0, switch_timestep - 20):switch_timestep]
    baseline_error = np.mean(pre_switch_window)
    
    # 2. Time-to-recover
    recovery_threshold = baseline_error * threshold_multiplier
    post_switch_errors = errors[switch_timestep:]
    
    time_to_recover = None
    for i, err in enumerate(post_switch_errors):
        if err <= recovery_threshold:
            time_to_recover = i
            break
            
    if time_to_recover is None:
        time_to_recover = len(post_switch_errors) # Didn't recover in the remaining horizon
        
    # 3. Area Under the Recovery Curve (AURC)
    aurc = np.sum(post_switch_errors)
    
    return {
        "baseline_error": baseline_error,
        "recovery_threshold": recovery_threshold,
        "time_to_recover": time_to_recover,
        "aurc": aurc
    }

def plot_recovery_curve(errors, switch_timestep, metrics, filename="../figs/recovery_curve.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(errors, label="Prediction Error (MSE)", color="royalblue", linewidth=2)
    plt.axvline(x=switch_timestep, color="crimson", linestyle="--", label="Dynamics Switch (A -> B)")
    
    # Draw baseline and threshold
    plt.axhline(y=metrics["baseline_error"], color="green", linestyle=":", label="Pre-switch Baseline")
    plt.axhline(y=metrics["recovery_threshold"], color="orange", linestyle=":", label="Recovery Threshold (110%)")
    
    plt.title("BDH Prediction Error Under Mid-Rollout Dynamics Switch")
    plt.xlabel("Timestep")
    plt.ylabel("Next-State Prediction MSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Recovery curve plot saved to {filename}")

if __name__ == "__main__":
    checkpoint_path = "../checkpoints/bdh_control_best.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint '{checkpoint_path}' not found. Please train the model first.")
        exit(1)
        
    print(f"Loading model checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = bdh.BDH(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Run rollout and switch
    errors, switch_time = run_switch_rollout(model, horizon=200, switch_timestep=100, regime_b="B")
    
    # Calculate metrics
    metrics = calculate_recovery_metrics(errors, switch_time)
    
    print("\n--- Evaluation Metrics ---")
    print(f"Pre-switch Baseline MSE:        {metrics['baseline_error']:.6f}")
    print(f"Recovery Threshold (110%):      {metrics['recovery_threshold']:.6f}")
    print(f"Time-to-Recover (steps):        {metrics['time_to_recover']}")
    print(f"Area Under Recovery Curve (AUC): {metrics['aurc']:.4f}")
    
    # Plot
    plot_recovery_curve(errors, switch_time, metrics)
