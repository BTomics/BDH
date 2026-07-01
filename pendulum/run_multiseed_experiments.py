import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import os
import torch
import torch.nn as nn
import numpy as np
import bdh
import baselines
from env import PendulumRegimeEnv

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
SEEDS = [42, 43, 44, 45, 46]
EPOCHS = 800  # Slightly fewer epochs to make multi-seed runs faster
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HORIZON = 200
SWITCH_TIMESTEP = 100

# Define Regimes
# Regime A: Standard
# Regime B_seen: Double gravity (seen/unseen variants can be configured)
# Regime B_novel: Double pole mass, standard gravity (completely different type of shift)
regime_variants = {
    "seen": "B",       # B has g=20.0, m=1.0, l=1.0
    "novel": "B_novel" # Defined in env.py regimes: g=10.0, m=3.0, l=1.0
}

# env.py already defines B_novel; override here as a safeguard in case it is missing
PendulumRegimeEnv.regimes = {
    "A": {"g": 10.0, "m": 1.0, "l": 1.0},
    "B": {"g": 20.0, "m": 1.0, "l": 1.0},
    "B_novel": {"g": 10.0, "m": 3.0, "l": 1.0}
}

# Load datasets
if not os.path.exists("../data/train_data.npz") or not os.path.exists("../data/val_data.npz"):
    raise FileNotFoundError("Datasets not found. Please run collect_data.py first.")

train_dataset = np.load("../data/train_data.npz")
val_dataset = np.load("../data/val_data.npz")

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

def evaluate_model_on_switch(model, model_name, regime_b, freeze_timestep=None):
    model.eval()
    env = PendulumRegimeEnv()
    env.set_regime("A")
    obs, info = env.reset(seed=42)
    
    history_states = torch.tensor([obs], dtype=torch.float32, device=device).unsqueeze(0)
    history_actions = torch.zeros((1, 0, 1), dtype=torch.float32, device=device)
    errors = []
    
    for t in range(HORIZON):
        if t == SWITCH_TIMESTEP:
            env.set_regime(regime_b)
            
        action_val = np.sin(t / 10.0) * 2.0
        action = np.array([action_val], dtype=np.float32)
        obs_next, reward, terminated, truncated, info = env.step(action)
        
        action_tensor = torch.tensor([action], dtype=torch.float32, device=device).unsqueeze(0)
        history_actions = torch.cat([history_actions, action_tensor], dim=1)
        
        with torch.no_grad():
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

# Initialize structures to hold multi-seed results
# Key structure: {model_name: {regime_type: {"t_rec": [], "aurc": []}}}
all_results = {
    "BDH (Live-sigma)": {"seen": {"t_rec": [], "aurc": []}, "novel": {"t_rec": [], "aurc": []}},
    "BDH (Frozen-sigma)": {"seen": {"t_rec": [], "aurc": []}, "novel": {"t_rec": [], "aurc": []}},
    "Transformer": {"seen": {"t_rec": [], "aurc": []}, "novel": {"t_rec": [], "aurc": []}},
    "GRU": {"seen": {"t_rec": [], "aurc": []}, "novel": {"t_rec": [], "aurc": []}},
    "MLP": {"seen": {"t_rec": [], "aurc": []}, "novel": {"t_rec": [], "aurc": []}}
}

# Run experiments across seeds
for seed in SEEDS:
    print(f"\n--- RUNNING SEED {seed} ---")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Define models
    bdh_config = bdh.BDHConfig(
        n_layer=3, n_embd=64, n_head=4, mlp_internal_dim_multiplier=8,
        state_dim=3, action_dim=1
    )
    
    models = {
        "BDH (Live-sigma)": bdh.BDH(bdh_config).to(device),
        "Transformer": baselines.TransformerModel(state_dim=3, action_dim=1, n_embd=96, n_head=4, n_layer=3).to(device),
        "GRU": baselines.GRUModel(state_dim=3, action_dim=1, n_embd=128, n_layer=2).to(device),
        "MLP": baselines.MLPModel(state_dim=3, action_dim=1, n_embd=280).to(device)
    }
    
    # Train models
    for name, model in models.items():
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
                    
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        
        # Evaluate this seed on both seen and novel switch regimes
        for regime_type, regime_b in regime_variants.items():
            # Live-sigma BDH
            if name == "BDH (Live-sigma)":
                errors = evaluate_model_on_switch(model, name, regime_b, freeze_timestep=None)
                base, t_rec, aurc = calculate_metrics(errors)
                all_results["BDH (Live-sigma)"][regime_type]["t_rec"].append(t_rec)
                all_results["BDH (Live-sigma)"][regime_type]["aurc"].append(aurc)
                
                # Frozen-sigma ablation
                errors_frozen = evaluate_model_on_switch(model, "BDH (Frozen-sigma)", regime_b, freeze_timestep=SWITCH_TIMESTEP)
                _, t_rec_f, aurc_f = calculate_metrics(errors_frozen)
                all_results["BDH (Frozen-sigma)"][regime_type]["t_rec"].append(t_rec_f)
                all_results["BDH (Frozen-sigma)"][regime_type]["aurc"].append(aurc_f)
            else:
                errors = evaluate_model_on_switch(model, name, regime_b)
                base, t_rec, aurc = calculate_metrics(errors)
                all_results[name][regime_type]["t_rec"].append(t_rec)
                all_results[name][regime_type]["aurc"].append(aurc)

# Print aggregated results (mean ± std)
print("\n" + "="*80)
print(f"{'Model':<18} | {'Seen Regime B (Time)':<22} | {'Novel Regime B (Time)':<22}")
print("="*80)
for name in all_results.keys():
    seen_t = all_results[name]["seen"]["t_rec"]
    novel_t = all_results[name]["novel"]["t_rec"]
    print(f"{name:<18} | {np.mean(seen_t):5.1f} ± {np.std(seen_t):4.1f} steps     | {np.mean(novel_t):5.1f} ± {np.std(novel_t):4.1f} steps")
print("="*80)

print("\n" + "="*80)
print(f"{'Model':<18} | {'Seen Regime B (AURC)':<22} | {'Novel Regime B (AURC)':<22}")
print("="*80)
for name in all_results.keys():
    seen_a = all_results[name]["seen"]["aurc"]
    novel_a = all_results[name]["novel"]["aurc"]
    print(f"{name:<18} | {np.mean(seen_a):6.2f} ± {np.std(seen_a):5.2f}          | {np.mean(novel_a):6.2f} ± {np.std(novel_a):5.2f}")
print("="*80)
