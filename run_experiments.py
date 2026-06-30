import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import bdh
import baselines
from env_wrapper import PendulumRegimeEnv

# Set device and seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

print(f"Using device: {device}")

# Experiment Parameters
EPOCHS = 1000
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HORIZON = 200
SWITCH_TIMESTEP = 100

# 1. Parameter Matching Configs
# We target around 150K to 250K parameters for all models.
bdh_config = bdh.BDHConfig(
    n_layer=3,
    n_embd=64,
    n_head=4,
    mlp_internal_dim_multiplier=8, # N = 8 * 64 // 4 = 128
    state_dim=3,
    action_dim=1
)

# Instantiate models
models = {
    "BDH (Live-sigma)": bdh.BDH(bdh_config).to(device),
    "Transformer": baselines.TransformerModel(
        state_dim=3, action_dim=1, n_embd=96, n_head=4, n_layer=3
    ).to(device),
    "GRU": baselines.GRUModel(
        state_dim=3, action_dim=1, n_embd=128, n_layer=2
    ).to(device),
    "MLP": baselines.MLPModel(
        state_dim=3, action_dim=1, n_embd=280
    ).to(device)
}

# Print parameter counts (PRO-90)
print("\n=== Model Parameter Matching (PRO-90) ===")
for name, model in models.items():
    print(f"{name:<15} : {baselines.get_param_count(model):,}")
print("=========================================\n")

# Load datasets
if not os.path.exists("train_data.npz") or not os.path.exists("val_data.npz"):
    raise FileNotFoundError("Datasets not found. Please run collect_data.py first.")

train_dataset = np.load("train_data.npz")
val_dataset = np.load("val_data.npz")

def get_batch(split, batch_size):
    dataset = train_dataset if split == "train" else val_dataset
    states = dataset["states"]
    actions = dataset["actions"]
    N = states.shape[0]
    ix = np.random.randint(0, N, batch_size)
    
    s_batch = torch.tensor(states[ix, :-1, :], dtype=torch.float32, device=device)
    a_batch = torch.tensor(actions[ix, :, :], dtype=torch.float32, device=device)
    target_batch = torch.tensor(states[ix, 1:, :], dtype=torch.float32, device=device)
    return s_batch, a_batch, target_batch

# 2. Train all models
trained_checkpoints = {}

for name, model in models.items():
    print(f"Training {name}...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    best_val_loss = float("inf")
    best_state = None
    
    for epoch in range(EPOCHS):
        model.train()
        s_batch, a_batch, target_batch = get_batch("train", BATCH_SIZE)
        
        preds, loss = model(s_batch, a_batch, target_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                s_val, a_val, target_val = get_batch("val", BATCH_SIZE)
                _, val_loss = model(s_val, a_val, target_val)
            
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                
    print(f"{name} training finished. Best Val MSE: {best_val_loss:.6f}")
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    trained_checkpoints[name] = model

# 3. Evaluate on dynamics switch (A -> B)
def evaluate_model_on_switch(model, model_name, freeze_timestep=None):
    """
    Runs a rollout on the environment, switching A -> B at SWITCH_TIMESTEP,
    and returns the prediction errors at each timestep.
    """
    model.eval()
    env = PendulumRegimeEnv()
    env.set_regime("A")
    
    obs, info = env.reset(seed=42)
    
    history_states = torch.tensor([obs], dtype=torch.float32, device=device).unsqueeze(0)
    history_actions = torch.zeros((1, 0, 1), dtype=torch.float32, device=device)
    
    errors = []
    
    for t in range(HORIZON):
        if t == SWITCH_TIMESTEP:
            env.set_regime("B")
            
        action_val = np.sin(t / 10.0) * 2.0
        action = np.array([action_val], dtype=np.float32)
        
        obs_next, reward, terminated, truncated, info = env.step(action)
        
        action_tensor = torch.tensor([action], dtype=torch.float32, device=device).unsqueeze(0)
        history_actions = torch.cat([history_actions, action_tensor], dim=1)
        
        with torch.no_grad():
            # Pass freeze_timestep to BDH if applicable
            if "BDH" in model_name:
                preds, _ = model(history_states, history_actions, freeze_timestep=freeze_timestep)
            else:
                preds, _ = model(history_states, history_actions)
                
            pred_next = preds[:, -1, :].cpu().numpy()[0]
            
        error = np.mean((obs_next - pred_next) ** 2)
        errors.append(error)
        
        obs_next_tensor = torch.tensor([obs_next], dtype=torch.float32, device=device).unsqueeze(0)
        history_states = torch.cat([history_states, obs_next_tensor], dim=1)
        
    env.close()
    return np.array(errors)

print("\nEvaluating all models on the dynamics switch...")
results = {}

# Evaluate Live-sigma BDH
results["BDH (Live-sigma)"] = evaluate_model_on_switch(trained_checkpoints["BDH (Live-sigma)"], "BDH (Live-sigma)", freeze_timestep=None)

# Evaluate Frozen-sigma BDH (Ablation - PRO-87)
results["BDH (Frozen-sigma)"] = evaluate_model_on_switch(trained_checkpoints["BDH (Live-sigma)"], "BDH (Frozen-sigma)", freeze_timestep=SWITCH_TIMESTEP)

# Evaluate Baselines
results["Transformer"] = evaluate_model_on_switch(trained_checkpoints["Transformer"], "Transformer")
results["GRU"] = evaluate_model_on_switch(trained_checkpoints["GRU"], "GRU")
results["MLP"] = evaluate_model_on_switch(trained_checkpoints["MLP"], "MLP")

# 4. Calculate Recovery Metrics (PRO-85)
def calculate_metrics(errors):
    pre_switch = errors[max(0, SWITCH_TIMESTEP - 20):SWITCH_TIMESTEP]
    baseline = np.mean(pre_switch)
    threshold = baseline * 1.1
    post_switch = errors[SWITCH_TIMESTEP:]
    
    time_to_recover = None
    for i, err in enumerate(post_switch):
        if err <= threshold:
            time_to_recover = i
            break
    if time_to_recover is None:
        time_to_recover = len(post_switch)
        
    aurc = np.sum(post_switch)
    return baseline, time_to_recover, aurc

print("\n=== EXPERIMENT RESULTS ===")
print(f"{'Model':<18} | {'Baseline MSE':<12} | {'Recovery Time':<13} | {'AURC':<10}")
print("-" * 62)

metrics_data = {}
for name, errors in results.items():
    base, t_rec, aurc = calculate_metrics(errors)
    metrics_data[name] = {"baseline": base, "t_rec": t_rec, "aurc": aurc}
    print(f"{name:<18} | {base:.6f}     | {t_rec:<13} | {aurc:.4f}")
print("==========================\n")

# 5. Plot Comparison Figure (PRO-93)
os.makedirs("figs", exist_ok=True)
plt.figure(figsize=(12, 6))

colors = {
    "BDH (Live-sigma)": "royalblue",
    "BDH (Frozen-sigma)": "dodgerblue",
    "Transformer": "forestgreen",
    "GRU": "darkorange",
    "MLP": "crimson"
}

styles = {
    "BDH (Live-sigma)": "-",
    "BDH (Frozen-sigma)": "--",
    "Transformer": "-",
    "GRU": "-",
    "MLP": "-"
}

for name, errors in results.items():
    plt.plot(errors, label=name, color=colors[name], linestyle=styles[name], linewidth=2)

plt.axvline(x=SWITCH_TIMESTEP, color="black", linestyle=":", label="Dynamics Switch (A -> B)")
plt.title("Prediction Error Recovery Under Mid-Rollout Dynamics Switch")
plt.xlabel("Timestep")
plt.ylabel("Next-State Prediction MSE")
plt.yscale("log") # Log scale helps visualize differences across orders of magnitude
plt.legend()
plt.grid(True, which="both", alpha=0.3)

plt.savefig("figs/all_recovery_curves.png", dpi=300)
plt.close()
print("Comparison plot saved to 'figs/all_recovery_curves.png'")
