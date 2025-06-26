import socket
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces

# Load config
with open("input_parameters.json", "r") as f:
    PARAMS = json.load(f)

DEBUG = PARAMS.get("DEBUG", False)
EVAL_MODE = PARAMS.get("evaluation", False)
CREATE_NEW = PARAMS.get("create_new_model", True)
LOG_DIR = PARAMS["log_dir_template"].format(datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(LOG_DIR, exist_ok=True)

ACTION_MIN = PARAMS.get("action_min", 10.0)
ACTION_MAX = PARAMS.get("action_max", 1000.0)

# UDP sockets
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind((PARAMS["hp_ip"], PARAMS["udp_port_recv"]))
print(f"Listening on {PARAMS['hp_ip']}:{PARAMS['udp_port_recv']}")

class CRIOUDPEnv(gym.Env):
    """
    Gym environment interfacing with UDP-driven hardware.
    - Observation: 4 floats (obs1–obs4).
    - Reward: obs2 (second observed value).
    - Action: scalar in [action_min, action_max], used to multiply obs1.
    """

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=ACTION_MIN, high=ACTION_MAX, shape=(1,), dtype=np.float32)
        self.timestamp = 0
        self.step_count = 0
        self.last_obs = np.zeros(4, dtype=np.float32)

    def reset(self):
        self.step_count = 0
        return self._recv_obs()

    def step(self, action):
        raw = float(action[0])
        clipped = float(np.clip(raw, ACTION_MIN, ACTION_MAX))
        if DEBUG:
            print(f"[DEBUG] Raw action={raw:.4f}, Clipped={clipped:.4f}")

        output = self.last_obs[0] * clipped
        sock_send.sendto(f"{self.timestamp};{output:.4f}".encode(),
                         (PARAMS["crio_ip"], PARAMS["udp_port_send"]))

        obs = self._recv_obs()
        reward = obs[1]
        self.step_count += 1
        done = self.step_count >= PARAMS["episode_length"]

        return obs, reward, done, {}

    def _recv_obs(self):
        while True:
            data, _ = sock_recv.recvfrom(1024)
            parts = data.decode().strip().split(";")
            if len(parts) != 5:
                if DEBUG: print(f"Malformed: {data!r}")
                continue

            self.timestamp = int(parts[0])
            obs = np.array([float(p) for p in parts[1:5]], dtype=np.float32)
            self.last_obs = obs
            if DEBUG:
                print(f"Received obs: {obs}, ts={self.timestamp}")
            return obs

    def render(self, mode="human"): pass
    def close(self): pass

env = DummyVecEnv([lambda: CRIOUDPEnv()])

MODEL_PATH = os.path.join(LOG_DIR, "ppo_crio")
if CREATE_NEW or not os.path.exists(MODEL_PATH + ".zip"):
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=PARAMS.get("ppo_learning_rate", 1e-3),
        tensorboard_log=LOG_DIR,
        policy_kwargs=dict(net_arch=[PARAMS["hidden_units"]] * 2),
    )
else:
    model = PPO.load(MODEL_PATH, env=env)
    print("Loaded existing model.")

if EVAL_MODE:
    rewards = []
    for ep in range(PARAMS["total_episodes"]):
        obs = env.reset()
        total = 0.0
        for _ in range(PARAMS["episode_length"]):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total += reward
        rewards.append(total)
        print(f"Episode {ep+1} reward: {total:.2f}")
    plt.plot(rewards)
    plt.title("Evaluation Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(LOG_DIR, "eval_rewards.png"))

else:
    model.learn(total_timesteps=PARAMS["total_episodes"] * PARAMS["episode_length"])
    model.save(MODEL_PATH)

sock_send.close()
sock_recv.close()
print("Run complete. Logs saved to", LOG_DIR)

