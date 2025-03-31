import time
from typing import Optional

# installed
import numpy as np
from gymnasium import Env
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Box

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


class MyEnv(Env):
    def __init__(self):
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(5,))
        self.action_space = Box(low=-1.0, high=1.0, shape=tuple())
        self.last_obs_time = None

    def step(self, action: ActType) -> (ObsType, float, float, bool):
        if self.last_obs_time is not None:
            elapsed_time = time.time() - self.last_obs_time
            print(f"{elapsed_time:.6f} seconds")
        
        # Capture the current time for the next step
        self.last_obs_time = time.time()
        
        return (np.array([0, 0, 0, 0, 0], dtype=np.float32), 1, True, False, {})

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> tuple[ObsType, dict[str, any]]:
        self.last_obs_time = time.time()  # Start timer on reset
        return (np.array([0.0, 0.0, 0.0, 0, 0], dtype=np.float32), {})


env = MyEnv()
check_env(env)  # does not output any warnings

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)

vec_env = model.get_env()
obs = vec_env.reset()
model.predict(obs)

# Reset environment before evaluation
obs = vec_env.reset()
done = False

# Track time
start_time = time.time()
done2=False
ii=1
steps_eval = 100

while not done2:
    step_start_time = time.time()  # Start timer for action
    action, _states = model.predict(obs, deterministic=True)  # Ensure deterministic actions
    obs, reward, done, _, _ = env.step(action)
    step_end_time = time.time()  # End timer

    if ii==steps_eval: done2=True

    ii = ii+1
    print(f"Time taken for step: {step_end_time - step_start_time:.6f} seconds")

# Total evaluation time
end_time = time.time()
print(f"Total evaluation time: {end_time - start_time:.6f} seconds")
