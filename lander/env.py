import gymnasium as gym
import numpy as np

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
