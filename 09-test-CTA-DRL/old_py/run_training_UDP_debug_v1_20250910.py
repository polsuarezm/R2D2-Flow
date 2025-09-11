# === Full Training Script with Enhanced TensorBoard Logging and EvalCallback ===
import socket, json, os, glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.logger import configure

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--json_file", required=True)

args = parser.parse_args()

print(f"running case: {args.json_file}")


# === Custom TensorBoard Logging Callback ===
class TensorboardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        base_env = self.training_env.envs[0].env  # unwrap Monitor
        self.logger.record("custom/reward", base_env.last_reward)
        self.logger.record("custom/action", base_env.last_action)
        self.logger.record("custom/step_count", base_env.step_count)
        return True

# === Load configuration ===
with open(f"./conf/{args.json_file}", "r") as f:
    PARAMS = json.load(f)

DEBUG = PARAMS.get("DEBUG", False)
DEBUG_IP = PARAMS.get("debugging_IP", False)
EVAL_MODE = PARAMS.get("evaluation", False)
CREATE_NEW = PARAMS.get("create_new_model", True)
ALGO_TYPE = PARAMS.get("algo_type", "PPO").upper()

MODEL_PATH = PARAMS.get("load_model_path", "")

LOG_DIR = PARAMS["log_dir_template"].format(datetime.now().strftime("%Y%m%d-%H%M"))
os.makedirs(LOG_DIR, exist_ok=True)

ACTION_MIN = float(PARAMS["action_min"])
ACTION_MAX = float(PARAMS["action_max"])
N_STEPS = int(PARAMS["n_steps"])
BATCH_SIZE = int(PARAMS["batch_size"])
N_EPOCHS = int(PARAMS["n_epochs"])
N_OBS_ARRAY_PER_UDP = int(PARAMS["size_obs_array_per_UDP"])
N_ACTUATOR_ARRAY = int(PARAMS["size_actuator_array"])

MESSAGE_TYPE = int(PARAMS["message_type"])
SCALAR_REW = float(PARAMS["scalar_reward"])
TOTAL_DESCARTE = int(PARAMS["total_descarte"])
TOTAL_DESCARTE_USED = int(PARAMS["total_descarte_used"])
N_OBS_ARRAY = N_OBS_ARRAY_PER_UDP * (TOTAL_DESCARTE_USED+1)

REWARD_TYPE = PARAMS.get("reward_type", "").upper()

# === UDP setup ===
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.setblocking(False)

recv_ip = PARAMS["debug_ip"] if DEBUG_IP else PARAMS["hp_ip"]
send_ip = PARAMS["debug_ip"] if DEBUG_IP else PARAMS["crio_ip"]

sock_recv.bind((recv_ip, PARAMS["udp_port_recv"]))
print(f"Listening on {recv_ip}:{PARAMS['udp_port_recv']}")

# === Custom Gym Environment ===
class CRIOUDPEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, (N_OBS_ARRAY,), dtype=np.float32)
        self.action_space = spaces.Box(ACTION_MIN, ACTION_MAX, (N_ACTUATOR_ARRAY,), dtype=np.float32)
        self.timestamp = 0
        self.step_count = 0
        self.global_step = 0
        self.last_obs = np.zeros(N_OBS_ARRAY, dtype=np.float32)
        self.last_reward = 0.0
        self.last_action = 0.0

        if os.path.exists("./csv_log/live_rewards_temp.csv"):
            os.remove("./csv_log/live_rewards_temp.csv")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        obs = np.ones(N_OBS_ARRAY) if EVAL_MODE else self._receive_observation()
        return obs, {}

    def step(self, action):
        raw_action = action if isinstance(action, (list, np.ndarray)) else [action]

        if MESSAGE_TYPE == 1:
            message = f"{self.timestamp};1;1;1;1;1;1;" + ';'.join(map(str, raw_action))
        else:
            message = f"{self.timestamp};{''.join(map(str, raw_action))}"

        sock_send.sendto(message.encode(), (send_ip, PARAMS["udp_port_send"]))
        obs = self._receive_observation()

        reward = self._compute_reward_peak_fft_v1(obs) if REWARD_TYPE == "CTA" else self._compute_reward_debug_internalUDP(obs)

        if self.step_count % 10 == 0:
            print(f"| rew = {reward:.4f} | action = {action} | step = {self.step_count}")

        action_val = action[0] if isinstance(action, (list, np.ndarray)) else float(action)
        self.last_action = action_val
        self.last_reward = reward

        with open(os.path.join(LOG_DIR, "live_rewards.csv"), "a") as archive_file, \
             open("./csv_log/live_rewards_temp.csv", "a") as tmp_file:
            self.global_step += 1
            archive_file.write(f"{self.global_step},{reward},{action_val},{self.timestamp},{obs[-4]},{obs[-3]},{obs[-2]},{obs[-1]}\n")
            tmp_file.write(f"{self.global_step},{reward},{action_val},{self.timestamp},{obs[-4]},{obs[-3]},{obs[-2]},{obs[-1]}\n")

        self.step_count += 1
        terminated = self.step_count >= PARAMS["episode_length"]
        return obs, reward, terminated, False, {}

    def _receive_observation(self):
        sock_recv.setblocking(False)
        while True:
            try:
                sock_recv.recvfrom(1024)
            except BlockingIOError:
                break
        sock_recv.setblocking(True)

        accumulated_parts = np.zeros(N_OBS_ARRAY_PER_UDP * (TOTAL_DESCARTE + 1), dtype=np.float32)
        for i in range(TOTAL_DESCARTE + 1):
            data, _ = sock_recv.recvfrom(1024)
            parts = data.decode().strip().split(";")
            aux_obs = np.array([float(x) for x in parts[1:5]], dtype=np.float32)
            start = i * N_OBS_ARRAY_PER_UDP
            accumulated_parts[start:start + N_OBS_ARRAY_PER_UDP] = aux_obs

        sock_recv.setblocking(False)
        self.timestamp = int(parts[0])
        self.last_obs = accumulated_parts[:N_OBS_ARRAY]
        return self.last_obs

    def _compute_reward_peak_fft_v1(self, obs_pre_reward):
        aux = obs_pre_reward[-2]
        return 1 - aux / SCALAR_REW

    def _compute_reward_debug_internalUDP(self, obs_pre_reward):
        return obs_pre_reward[-1]

    def render(self, mode="human"): pass
    def close(self): pass

# === Environment setup ===
def make_env():
    env = CRIOUDPEnv()
    return Monitor(env, filename=os.path.join(LOG_DIR, "env_monitor"))

env = DummyVecEnv([make_env])

# === Load or Train Model ===
model = None
model_name = f"{ALGO_TYPE.lower()}_crio"

if CREATE_NEW or not os.path.exists(MODEL_PATH + ".zip"):

    eval_callback = EvalCallback(
    env,
    best_model_save_path=LOG_DIR,
    log_path=LOG_DIR,
    eval_freq=PARAMS.get("eval_freq", 5000),
    n_eval_episodes=PARAMS.get("n_eval_episodes", 1),  # <--- added
    deterministic=True,
    render=False
    )
    callback = CallbackList([TensorboardLoggingCallback(), eval_callback])

    if ALGO_TYPE == "PPO":
        model = PPO(
            "MlpPolicy", env, verbose=1,
            learning_rate=PARAMS.get("ppo_learning_rate", 1e-3),
            device="cpu",
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=PARAMS.get("ppo_gamma", 0.99),
            tensorboard_log=LOG_DIR,
            policy_kwargs=dict(
                net_arch=dict(
                    pi=PARAMS.get("actor_layers", [8]),
                    vf=PARAMS.get("critic_layers", [16, 64, 64])),
                log_std_init=PARAMS.get("ppo_log_std_init", -0.5),
            ),
        )
    else:
        raise NotImplementedError("Only PPO supported in this script version.")

    for i in range(int(PARAMS["total_episodes"] * PARAMS["episode_length"] // N_STEPS)):
        model.learn(total_timesteps=N_STEPS, reset_num_timesteps=False, callback=callback)
        model.save(os.path.join(LOG_DIR, f"model_{ALGO_TYPE}_{datetime.now().strftime('%H%M%S')}"))

else:
    model = PPO.load(MODEL_PATH, env=env)
    print(f"Loaded existing {ALGO_TYPE} model from: {MODEL_PATH}.zip")

# === Training monitor ===
monitor_files = glob.glob(os.path.join(LOG_DIR, "*monitor.csv"))
if monitor_files:
    df = pd.read_csv(monitor_files[0], skiprows=1)
    if 'r' in df.columns and 't' in df.columns:
        plt.figure(figsize=(8, 4))
        plt.plot(df["t"], df["r"], marker='o', linestyle='None')
        plt.xlabel("Timesteps")
        plt.ylabel("Episode Reward")
        plt.title(f"Training Progress - {ALGO_TYPE}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(LOG_DIR, "reward_vs_steps.png"))

sock_send.close()
sock_recv.close()
print("Execution complete. Logs saved in:", LOG_DIR)
