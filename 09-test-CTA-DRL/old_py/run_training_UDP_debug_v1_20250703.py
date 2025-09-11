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

# === Load configuration ===
with open("input_parameters_v1_20250703_debugip.json", "r") as f:
    PARAMS = json.load(f)

DEBUG = PARAMS.get("DEBUG", False)
DEBUG_IP = PARAMS.get("debugging_IP", False)
EVAL_MODE = PARAMS.get("evaluation", False)
CREATE_NEW = PARAMS.get("create_new_model", True)
ALGO_TYPE = PARAMS.get("algo_type", "PPO").upper()

model_path = PARAMS.get("load_model_path", "")
LOG_DIR = PARAMS["log_dir_template"].format(datetime.now().strftime("%Y%m%d-%H%M"))
os.makedirs(LOG_DIR, exist_ok=True)

ACTION_MIN = float(PARAMS["action_min"])
ACTION_MAX = float(PARAMS["action_max"])
N_STEPS = int(PARAMS["n_steps"])
BATCH_SIZE = int(PARAMS["batch_size"])
N_EPOCHS = int(PARAMS["n_epochs"])
N_OBS_ARRAY = int(PARAMS["size_obs_array"])
N_ACTUATOR_ARRAY = int(PARAMS["size_actuator_array"])
MESSAGE_TYPE = int(PARAMS["message_type"])
SCALAR_REW = float(PARAMS["scalar_reward"])
TOTAL_DESCARTE = int(PARAMS["total_descarte"])

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

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        if EVAL_MODE:
            obs = np.ones(N_OBS_ARRAY)
        else:
            obs, _ = self._receive_observation()
        return obs, {}

    def step(self, action):
        raw_action = action if isinstance(action, (list, np.ndarray)) else [action]

        if MESSAGE_TYPE == 1:
            message = f"{self.timestamp};1;1;1;1;1;1;" + ';'.join(map(str, raw_action))
        else:
            message = f"{self.timestamp};{''.join(map(str, raw_action))}"

        sock_send.sendto(message.encode(), (send_ip, PARAMS["udp_port_send"]))
        obs, aux_obs = self._receive_observation()
        reward = obs[3]

        if self.step_count % 10 == 0:
            print(f"| rew = {reward:.4f} | action = {action} | step = {self.step_count}")

        with open(os.path.join(LOG_DIR, "live_rewards.csv"), "a") as f:
            self.global_step += 1
            #f.write(f"{self.global_step},{obs[0]},{obs[1]},{obs[2]},{obs[3]},{reward},{action[0]},{self.timestamp}\n")
            action_val = action[0] if isinstance(action, (list, np.ndarray)) else float(action)
            f.write(f"{self.global_step},{obs[0]},{obs[1]},{obs[2]},{obs[3]},{reward},{action_val},{self.timestamp}\n")


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

        for _ in range(TOTAL_DESCARTE + 1):
            data, _ = sock_recv.recvfrom(1024)
            parts = data.decode().strip().split(";")
            aux_obs = np.array([float(x) for x in parts[1:5]], dtype=np.float32)

        sock_recv.setblocking(False)
        self.timestamp = int(parts[0])
        self.last_obs = np.array([float(x) for x in parts[1:5]], dtype=np.float32)
        return self.last_obs, aux_obs[-1]

    def render(self, mode="human"): pass
    def close(self): pass

# === Create monitored env ===
def make_env():
    env = CRIOUDPEnv()
    return Monitor(env, filename=os.path.join(LOG_DIR, "env_monitor"))

env = DummyVecEnv([make_env])

# === Load or Train Model ===
model = None
model_name = f"{ALGO_TYPE.lower()}_crio"

if CREATE_NEW or not os.path.exists(model_path + ".zip"):
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
    elif ALGO_TYPE == "DDPG":
        n_actions = env.action_space.shape[0]
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=PARAMS.get("ou_sigma", 0.1) * np.ones(n_actions),
        )
        model = DDPG(
            "MlpPolicy", env, verbose=1,
            learning_rate=PARAMS.get("ddpg_learning_rate", 1e-3),
            buffer_size=PARAMS.get("buffer_size", 100000),
            batch_size=PARAMS.get("batch_size", 128),
            tau=PARAMS.get("tau", 0.005),
            gamma=PARAMS.get("gamma", 0.99),
            learning_starts=PARAMS.get("learning_starts", 100),
            train_freq=PARAMS.get("train_freq", (1, "step")),
            gradient_steps=PARAMS.get("gradient_steps", 1),
            action_noise=action_noise,
            tensorboard_log=LOG_DIR,
            policy_kwargs=dict(
                net_arch=dict(
                    pi=PARAMS.get("actor_layers", [64, 64]),
                    qf=PARAMS.get("critic_layers", [64, 64])),
            ),
        )
else:
    print(f"Loading model from: {model_path}.zip")
    model = PPO.load(model_path, env=env) if ALGO_TYPE == "PPO" else DDPG.load(model_path, env=env)
    print(f"Loaded existing {ALGO_TYPE} model from: {model_path}.zip")

# === Training or Evaluation ===
if not EVAL_MODE:
    for i in range(int(PARAMS["total_episodes"] * PARAMS["episode_length"] // N_STEPS)):
        model.learn(total_timesteps=N_STEPS, reset_num_timesteps=False)
        model.save(os.path.join(LOG_DIR, f"model_{ALGO_TYPE}_{datetime.now().strftime('%H%M%S')}"))

    # Post-training evaluation
    print("Training finished. Running 1 evaluation episode...")
    obs = env.reset()[0]
    done = False
    eval_rewards = []
    step_idx = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        #obs, reward, done, _, _ = env.step(action)
        obs, reward, done, info = env.step(action)
        eval_rewards.append((step_idx, reward))
        step_idx += 1

    # Save evaluation rewards
    eval_csv = os.path.join(LOG_DIR, "evaluation_rewards.csv")
    with open(eval_csv, "w") as f:
        f.write("step,reward\n")
        for step, rew in eval_rewards:
            f.write(f"{step},{rew}\n")
    print(f"Evaluation episode complete. Saved to {eval_csv}")

else:
    for ep in range(PARAMS["total_episodes"]):
        obs = env.reset()[0]
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)

# === Plot training monitor ===
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
        print("Saved: reward_vs_steps.png")

# === Plot evaluation rewards ===
if os.path.exists(eval_csv):
    df_eval = pd.read_csv(eval_csv)
    plt.figure(figsize=(8, 4))
    plt.plot(df_eval["step"], df_eval["reward"], marker="o")
    plt.title("Evaluation Episode Rewards")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "evaluation_rewards.png"))
    print("Saved: evaluation_rewards.png")

sock_send.close()
sock_recv.close()
print("Execution complete. Logs saved in:", LOG_DIR)
