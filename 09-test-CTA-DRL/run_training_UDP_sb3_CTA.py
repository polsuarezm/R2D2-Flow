import socket
import time
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque

import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# === Load config ===
with open("input_parameters.json", "r") as f:
    PARAMS = json.load(f)

DEBUG = PARAMS.get("DEBUG", False)
EVAL_MODE = PARAMS.get("evaluation", False)
LOG_DIR = PARAMS["log_dir_template"].format(datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(LOG_DIR, exist_ok=True)

# === Logging ===
log_buffer = deque(maxlen=1000)
def debug_log(msg, flush=False):
    if DEBUG:
        log_buffer.append(msg)
        if flush or len(log_buffer) >= 100:
            with open(os.path.join(LOG_DIR, "debug_log.txt"), "a") as f:
                f.write("\n".join(log_buffer) + "\n")
            log_buffer.clear()

# === UDP setup ===
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind((PARAMS["pc_ip"], PARAMS["udp_port_recv"]))
print(f"Listening on {PARAMS['pc_ip']}:{PARAMS['udp_port_recv']}")

# === Custom Gym environment ===
class CRIOUDPEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=20.0, high=1000.0, shape=(1,), dtype=np.float32)
        self.timestamp = 0
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        obs, _ = self._recv_obs_reward()
        return np.array([obs], dtype=np.float32)

    def step(self, action):
        action_val = int(np.clip(action[0], 20, 1000))
        hex_str = format(action_val, '016x')
        message = f"{hex_str};{self.timestamp}"
        sock_send.sendto(message.encode(), (PARAMS["crio_ip"], PARAMS["udp_port_send"]))

        obs, reward = self._recv_obs_reward()
        self.current_step += 1
        done = self.current_step >= PARAMS["episode_length"]
        return np.array([obs], dtype=np.float32), reward, done, {}

    def _recv_obs_reward(self):
        while True:
            try:
                data_rcv, _ = sock_recv.recvfrom(32)
                decoded = data_rcv.decode()
                parts = decoded.strip().split(";")
                if len(parts) < 3:
                    continue
                self.timestamp = int(parts[0])
                obs_val = float(parts[1])
                reward_val = float(parts[2])
                return obs_val, reward_val
            except Exception as e:
                debug_log(f"[UDP ERROR] {e}")
                continue

    def render(self, mode='human'):
        pass

    def close(self):
        pass

# === Initialize environment ===
env = DummyVecEnv([lambda: CRIOUDPEnv()])

# === Train or Evaluate ===
if EVAL_MODE:
    print("🔍 Evaluation mode")
    model = PPO.load(os.path.join(LOG_DIR, "ppo_crio"))
    rewards = []
    for ep in range(PARAMS["total_episodes"]):
        obs = env.reset()
        total_reward = 0
        for _ in range(PARAMS["episode_length"]):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]
        print(f"[EVAL] Episode {ep+1} reward: {total_reward:.2f}")
        rewards.append(total_reward)
    plt.plot(rewards)
    plt.title("Evaluation Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(LOG_DIR, "eval_rewards.png"))

else:
    print("🚀 Training mode")
    model = PPO(
        "MlpPolicy", env,
        verbose=1,
        learning_rate=PARAMS.get("ppo_learning_rate", 1e-3),
        tensorboard_log=LOG_DIR,
        policy_kwargs=dict(net_arch=[PARAMS["hidden_units"], PARAMS["hidden_units"]])
    )
    model.learn(total_timesteps=PARAMS["total_episodes"] * PARAMS["episode_length"])
    model.save(os.path.join(LOG_DIR, "ppo_crio"))

# === Cleanup ===
sock_send.close()
sock_recv.close()
print("✅ Done.")
