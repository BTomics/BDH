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
                "g": 20.0,  # Slightly increased gravity (survivable switch)
                "m": 1.0,
                "l": 1.0
            },
            "B_novel": {
                "g": 20.0,
                "m": 3.0,
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

class RandomizedPendulumEnv(gym.Env):
    """
    Pendulum environment where gravity is randomized on each reset.
    If privileged=True, the observation includes the current gravity value.
    """
    def __init__(self, render_mode=None, privileged=False):
        self.env = gym.make("Pendulum-v1", render_mode=render_mode)
        self.privileged = privileged
        
        obs_dim = 4 if privileged else 3
        # Original limits: cos/sin [-1, 1], dot [-8, 8]. Gravity [5, 25]
        high = np.array([1.0, 1.0, 8.0] + ([25.0] if privileged else []), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = self.env.action_space
        self.current_g = 10.0
        
    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.current_g = np.random.uniform(5.0, 25.0)
        self.env.unwrapped.g = self.current_g
        info['g'] = self.current_g
        
        if self.privileged:
            obs = np.append(obs, self.current_g).astype(np.float32)
            
        return obs, info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['g'] = self.current_g
        
        if self.privileged:
            obs = np.append(obs, self.current_g).astype(np.float32)
            
        return obs, reward, terminated, truncated, info
        
    def close(self):
        self.env.close()

class LunarLanderRegimeEnv:
    def __init__(self, render_mode=None, privileged=False):
        """
        privileged: If True, appends the thruster health status (1.0 for normal, 0.2 for broken) to the state.
        """
        self.env = gym.make("LunarLanderContinuous-v3", render_mode=render_mode)
        self.privileged = privileged
        self.regimes = {
            "A": {"health": 1.0}, # Normal
            "B": {"health": 0.2}  # Left thruster 80% failure
        }
        self.current_regime = "A"

    def set_regime(self, regime_name: str):
        if regime_name not in self.regimes:
            raise ValueError(f"Unknown regime: {regime_name}")
        self.current_regime = regime_name

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.privileged:
            obs = np.append(obs, self.regimes[self.current_regime]["health"]).astype(np.float32)
        return obs, info

    def step(self, action):
        # Action is [main_engine, side_engines]
        # side_engines: > 0.5 is left engine, < -0.5 is right engine.
        effective_action = np.copy(action)
        
        health = self.regimes[self.current_regime]["health"]
        if health < 1.0 and effective_action[1] > 0.5:
            # Simulate left thruster failure
            # Compress the action space above 0.5
            effective_action[1] = 0.5 + (effective_action[1] - 0.5) * health

        obs, reward, terminated, truncated, info = self.env.step(effective_action)
        
        if self.privileged:
            obs = np.append(obs, health).astype(np.float32)
            
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()

    @property
    def observation_space(self):
        if self.privileged:
            return gym.spaces.Box(
                low=np.append(self.env.observation_space.low, 0.0),
                high=np.append(self.env.observation_space.high, 1.0),
                dtype=np.float32
            )
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space
