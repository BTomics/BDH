import gymnasium as gym
import numpy as np

class PendulumRegimeEnv:
    def __init__(self, render_mode=None):
        # TODO: Initialize the 'Pendulum-v1' environment using gymnasium
        self.env = gym.make("Pendulum-v1", render_mode=render_mode)
        
        # Define the parameters for the regimes.
        # Regime A is the default Pendulum-v1 parameters.
        # Regime B has modified parameters (e.g., double gravity) to simulate altered dynamics.
        self.regimes = {
            "A": {
                "g": 10.0,
                "m": 1.0,
                "l": 1.0
            },
            "B": {
                "g": 20.0,  # Double gravity
                "m": 1.0,
                "l": 1.0
            }
        }
        
        # Set the default regime to A
        self.current_regime = None
        self.set_regime("A")

    def set_regime(self, regime_name: str):
        """
        Modifies the physical parameters of the underlying environment to match the chosen regime.
        """
        if regime_name not in self.regimes:
            raise ValueError(f"Unknown regime: {regime_name}")
        
        self.current_regime = regime_name
        params = self.regimes[regime_name]
        
        # TODO: Update the physics parameters on the unwrapped environment.
        # Hint: In Gymnasium, you can access the raw environment attributes using `self.env.unwrapped`.
        # You need to update:
        # - self.env.unwrapped.g
        # - self.env.unwrapped.m
        # - self.env.unwrapped.l
        self.env.unwrapped.g = params["g"]
        self.env.unwrapped.m = params["m"]
        self.env.unwrapped.l = params["l"]
        
        print(f"Switched to Regime {regime_name}: g={params['g']}, m={params['m']}, l={params['l']}")

    def reset(self, seed=None):
        # TODO: Reset the environment and return the initial observation and info
        # Hint: delegate to self.env.reset(seed=seed)
        return self.env.reset(seed=seed)

    def step(self, action):
        # TODO: Step the environment forward using the action
        # Return: observation, reward, terminated, truncated, info
        # Hint: delegate to self.env.step(action)
        return self.env.step(action)

    def close(self):
        if self.env is not None:
            self.env.close()

if __name__ == "__main__":
    print("Testing PendulumRegimeEnv...")
    env = PendulumRegimeEnv()

    # 1. Reset the environment
    obs, info = env.reset(seed=42)
    print(f"Initial observation: {obs}\n")

    action = np.array([1.0]) # Constant torque

    # 2. Step 5 times under Regime A
    print("--- Regime A ---")
    for i in range(5):
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: obs={obs}")

    # 3. Switch to Regime B
    env.set_regime("B")

    # 4. Step 5 times under Regime B
    print("--- Regime B ---")
    for i in range(5):
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: obs={obs}")
    
    env.close()

