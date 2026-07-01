import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import numpy as np
from env import PendulumRegimeEnv

def collect_rollouts(num_episodes: int, horizon: int, regime: str = "A") -> dict:
    """
    Collects rollouts in the specified regime using a random policy.
    
    Returns a dictionary containing:
        - 'states': np.ndarray of shape (num_episodes, horizon + 1, 3)
        - 'actions': np.ndarray of shape (num_episodes, horizon, 1)
    """
    env = PendulumRegimeEnv()
    env.set_regime(regime)
    
    # Pre-allocate arrays for efficiency
    all_states = np.zeros((num_episodes, horizon + 1, 3), dtype=np.float32)
    all_actions = np.zeros((num_episodes, horizon, 1), dtype=np.float32)
    
    print(f"Collecting {num_episodes} rollouts of horizon {horizon} in Regime {regime}...")
    
    for ep in range(num_episodes):
        # Reset the environment and store the initial state
        obs, info = env.reset()
        all_states[ep, 0] = obs
        
        for t in range(horizon):
            # Sample a random action from the environment's action space [-2.0, 2.0]
            action = env.env.action_space.sample()
            
            # Step the environment forward with the action
            obs_next, reward, terminated, truncated, info = env.step(action)
            
            # Store the action and the next state in the pre-allocated arrays
            all_actions[ep, t] = action
            all_states[ep, t + 1] = obs_next
            
            if terminated or truncated:
                break
                
    env.close()
    
    return {
        "states": all_states,
        "actions": all_actions
    }

if __name__ == "__main__":
    NUM_TRAIN_EPISODES = 100
    NUM_VAL_EPISODES = 20
    HORIZON = 200  # Default Pendulum-v1 episode length
    
    # 1. Collect training data in Regime A
    train_data = collect_rollouts(NUM_TRAIN_EPISODES, HORIZON, regime="A")
    
    # 2. Collect validation data in Regime A
    val_data = collect_rollouts(NUM_VAL_EPISODES, HORIZON, regime="A")
    
    # 3. Save the collected data to disk
    np.savez_compressed("../data/train_data.npz", states=train_data["states"], actions=train_data["actions"])
    np.savez_compressed("../data/val_data.npz", states=val_data["states"], actions=val_data["actions"])
    
    print("Datasets saved successfully.")
    
    # 4. Sanity check: load the saved files and print their shapes
    loaded_train = np.load("../data/train_data.npz")
    loaded_val = np.load("../data/val_data.npz")
    
    print("\n--- Sanity Check ---")
    print(f"Train States Shape:  {loaded_train['states'].shape} (Expected: (100, 201, 3))")
    print(f"Train Actions Shape: {loaded_train['actions'].shape} (Expected: (100, 200, 1))")
    print(f"Val States Shape:    {loaded_val['states'].shape} (Expected: (20, 201, 3))")
    print(f"Val Actions Shape:   {loaded_val['actions'].shape} (Expected: (20, 200, 1))")
