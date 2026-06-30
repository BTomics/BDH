import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import os
import torch
import numpy as np
import bdh

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

print(f"Using device: {device}")

# Hyperparameters
EPOCHS = 1500
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EVAL_FREQ = 100

# Load datasets
if not os.path.exists("../data/train_data.npz") or not os.path.exists("../data/val_data.npz"):
    raise FileNotFoundError("Datasets not found. Please run collect_data.py first.")

train_dataset = np.load("../data/train_data.npz")
val_dataset = np.load("../data/val_data.npz")

# Helper to get batches
def get_batch(split, batch_size):
    dataset = train_dataset if split == "train" else val_dataset
    states = dataset["states"]   # (N, H+1, 3)
    actions = dataset["actions"] # (N, H, 1)
    
    N = states.shape[0]
    ix = np.random.randint(0, N, batch_size)
    
    # Inputs: s_t, a_t
    # Targets: s_{t+1}
    s_batch = torch.tensor(states[ix, :-1, :], dtype=torch.float32, device=device)
    a_batch = torch.tensor(actions[ix, :, :], dtype=torch.float32, device=device)
    target_batch = torch.tensor(states[ix, 1:, :], dtype=torch.float32, device=device)
    
    return s_batch, a_batch, target_batch

# Initialize model
config = bdh.BDHConfig(
    n_layer=4,              # Slightly smaller for faster training on toy task
    n_embd=128,
    n_head=4,
    state_dim=3,
    action_dim=1
)
model = bdh.BDH(config).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

print("Starting training...")
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    s_batch, a_batch, target_batch = get_batch("train", BATCH_SIZE)
    
    # Forward pass
    predictions, loss = model(s_batch, a_batch, target_batch)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Periodic evaluation
    if epoch % EVAL_FREQ == 0 or epoch == EPOCHS - 1:
        model.eval()
        with torch.no_grad():
            # Get validation batch
            s_val, a_val, target_val = get_batch("val", BATCH_SIZE)
            val_preds, val_loss = model(s_val, a_val, target_val)
            
        print(f"Epoch {epoch:4d} | Train MSE: {loss.item():.6f} | Val MSE: {val_loss.item():.6f}")
        
        # Save best model checkpoint
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config
            }, "checkpoints/bdh_control_best.pt")

print(f"\nTraining completed. Best Val MSE: {best_val_loss:.6f}")
print("Checkpoint saved to 'checkpoints/bdh_control_best.pt'")
