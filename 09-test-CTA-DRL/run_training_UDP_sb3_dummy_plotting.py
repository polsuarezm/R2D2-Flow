import socket, json, os, glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results

# === Load configuration ===
with open("input_parameters.json", "r") as f:
    PARAMS = json.load(f)

DEBUG = PARAMS.get("DEBUG", False)
EVAL_MODE = PARAMS.get("evaluation", False)
CREATE_NEW = PARAMS.get("create_new_model", True)

LOG_DIR = PARAMS["log_dir_template"].format(datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(LOG_DIR, exist_ok=True)

ACTION_MIN = float(PARAMS["action_min"])
ACTION_MAX = float(PARAMS["action_max"])
N_STEPS = int(PARAMS["n_steps"])
BATCH_SIZE = int(PARAMS["batch_size"])
N_EPOCHS = int(PARAMS["n_epochs"])

# === UDP setup ===
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind((PARAMS["hp_ip"], PARAMS["udp_port_recv"]))
print(f"Listening on {PARAMS['hp_ip']}:{PARAMS['udp_port_recv']}")

# === Environment ===
class CRIOUDPEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32)
        self.action_space = spaces.Box(ACTION_MIN, ACTION_MAX, (1,), dtype=np.float32)
        self.timestamp = 0
        self.step_count = 0
        self.last_obs = np.zeros(4, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        obs = self._receive_observation()
        return obs, {}

    def step(self, action):
        raw = float(action[0])
        clipped = float(np.clip(raw, ACTION_MIN, ACTION_MAX))
        output = clipped * 500 + 500
        sock_send.sendto(
            f"{self.timestamp};{output:.4f}".encode(),
            (PARAMS["crio_ip"], PARAMS["udp_port_send"])
        )
        obs = self._receive_observation()
        reward = obs[1] - 4
        self.step_count += 1
        terminated = self.step_count >= PARAMS["episode_length"]
        truncated = False
        if DEBUG:
            print(f"Step {self.step_count} | Raw={raw:.4f}, Clipped={clipped:.4f}, Reward={reward:.4f}")
        return obs, reward, terminated, truncated, {}

    def _receive_observation(self):
        while True:
            data, _ = sock_recv.recvfrom(1024)
            parts = data.decode().strip().split(";")
            if len(parts) != 5:
                if DEBUG: print(f"Malformed packet: {data!r}")
                continue
            self.timestamp = int(parts[0])
            self.last_obs = np.array([float(x) for x in parts[1:5]], dtype=np.float32)
            return self.last_obs

    def render(self, mode="human"): pass
    def close(self): pass

# === Monitor + DummyVecEnv ===
def make_env():
    env = CRIOUDPEnv()
    # explicitly name the monitor file so that load_results can find it
    return Monitor(env, filename=os.path.join(LOG_DIR, "env_monitor.monitor.csv"))

env = DummyVecEnv([make_env])

# === PPO setup ===
model_path = os.path.join(LOG_DIR, "ppo_crio")
if CREATE_NEW or not os.path.exists(model_path + ".zip"):
    model = PPO(
        "MlpPolicy", env, verbose=1,
        learning_rate=PARAMS.get("ppo_learning_rate", 1e-3),
        n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS,
        tensorboard_log=LOG_DIR,
        policy_kwargs=dict(net_arch=[PARAMS["hidden_units"]] * 2),
    )
    print(f"PPO configured (n_steps={N_STEPS}, batch_size={BATCH_SIZE}, n_epochs={N_EPOCHS})")
else:
    model = PPO.load(model_path, env=env)
    print("Loaded existing model.")

# === Training or evaluation ===
if not EVAL_MODE:
    model.learn(total_timesteps=int(PARAMS["total_episodes"] * PARAMS["episode_length"]))
    model.save(model_path)

    # Verify monitor file exists
    files = glob.glob(os.path.join(LOG_DIR, "*monitor.csv"))
    print("Monitor files:", files)

    # Plot reward vs timesteps
    results = load_results(LOG_DIR)
    x, y = ts2xy(results, "timesteps")
    plot_results([results], LOG_DIR, "timesteps", "ppo_crio")
    plt.savefig(os.path.join(LOG_DIR, "reward_vs_steps.png"))
    print("Saved reward_vs_steps.png")

else:
    for ep in range(PARAMS["total_episodes"]):
        obs, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total += reward
        print(f"Eval Episode {ep+1}: Total Reward = {total:.2f}")

# === Cleanup ===
sock_send.close()
sock_recv.close()
print("Run completed. Find logs and plots in:", LOG_DIR)

