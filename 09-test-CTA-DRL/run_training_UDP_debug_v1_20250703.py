import socket, json, os, glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results

# === Load configuration ===
with open("input_parameters_v1_20250703.json", "r") as f:
    PARAMS = json.load(f)

DEBUG = PARAMS.get("DEBUG", False)
EVAL_MODE = PARAMS.get("evaluation", False)
CREATE_NEW = PARAMS.get("create_new_model", True)
ALGO_TYPE = PARAMS.get("algo_type", "PPO").upper()

model_path = "/scratch/polsm/011-DRL-experimental/AFC-DRL-experiment-v3/09-test-CTA-DRL/logs/DDPG_V1_20250702-141746"

LOG_DIR = PARAMS["log_dir_template"].format(datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(LOG_DIR, exist_ok=True)

ACTION_MIN = float(PARAMS["action_min"])
ACTION_MAX = float(PARAMS["action_max"])
N_STEPS = int(PARAMS["n_steps"])
BATCH_SIZE = int(PARAMS["batch_size"])
N_EPOCHS = int(PARAMS["n_epochs"])
N_OBS_ARRAY = int(PARAMS["size_obs_array"])
N_ACTUATOR_ARRAY = int(PARAMS["size_actuator_array"])
MESSAGE_TYPE = int(PARAMS["message_type"])  # 1 for array, 2 for string
SCALAR_REW = float(PARAMS["scalar_reward"])
TOTAL_DESCARTE = int(PARAMS["total_descarte"])


# === UDP setup ===
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

sock_recv.setblocking(False)
sock_recv.bind((PARAMS["crio_ip"], PARAMS["udp_port_recv"]))
print(f"Listening on {PARAMS['hp_ip']}:{PARAMS['udp_port_recv']}")

# === Environment ===
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
            obs = [1,1,1,1]
        else:
            obs, _ = self._receive_observation()
        return obs, {}

    def step(self, action):
        raw_action = action

        if MESSAGE_TYPE == 1:
            message = f"{self.timestamp};1;1;1;1;1;1;" + ';'.join(map(str, raw_action))
        else:
            message = f"{self.timestamp};{''.join(map(str, raw_action))}"

        sock_send.sendto(
            message.encode(),
            (PARAMS["hp_ip"], PARAMS["udp_port_send"])
        )

        obs, aux_obs = self._receive_observation()
        reward = obs[3]
        if self.step_count%10==0: 
            print(f"| rew = {reward:.4f} | action = {action} | step = {self.step_count}")

        with open(os.path.join(LOG_DIR, "live_rewards.csv"), "a") as f:
            self.global_step = self.global_step + 1
            f.write(f"{self.global_step},{obs[0]},{obs[1]},{obs[2]},{obs[3]},{reward},{action[0]},{self.timestamp}\n")

        self.step_count += 1
        terminated = self.step_count >= PARAMS["episode_length"]
        truncated = False

        if DEBUG:
            print("Received rew:", reward, "action used:", aux_obs)

        return obs, reward, terminated, truncated, {}

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
            #print(f"descarte: rew = {(SCALAR_REW - aux_obs[3]) / SCALAR_REW} / ts = {int(parts[0])}")
        sock_recv.setblocking(False)

        self.timestamp = int(parts[0])
        self.last_obs = np.array([float(x) for x in parts[1:5]], dtype=np.float32)

        return self.last_obs, aux_obs[-1]

    def render(self, mode="human"): pass
    def close(self): pass

# === Wrap environment with Monitor ===
def make_env():
    env = CRIOUDPEnv()
    return Monitor(env, filename=os.path.join(LOG_DIR, "env_monitor"))

env = DummyVecEnv([make_env])

# === Model setup ===
model_name = f"{ALGO_TYPE.lower()}_crio"
#model_path = os.path.join(LOG_DIR, model_name)

# === Choose algorithm ===
if CREATE_NEW or not os.path.exists(model_path + ".zip"):
    if ALGO_TYPE == "PPO":
        actor_layers = PARAMS.get("actor_layers", [8])
        critic_layers = PARAMS.get("critic_layers", [16, 64, 64])

        model = PPO(
            "MlpPolicy", env, verbose=1,
            learning_rate=PARAMS.get("ppo_learning_rate", 1e-3),
            device="cpu",
            n_steps=N_STEPS, 
            batch_size=BATCH_SIZE, 
            n_epochs=N_EPOCHS,
            tensorboard_log=LOG_DIR,
            policy_kwargs=dict(
                net_arch=dict(pi=actor_layers, vf=critic_layers)
            )
        )

    elif ALGO_TYPE == "DDPG":

        actor_layers = PARAMS.get("actor_layers", [64, 64])
        critic_layers = PARAMS.get("critic_layers", [64, 64])

        # Setup action noise
        n_actions = env.action_space.shape[0]
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=PARAMS.get("ou_sigma", 0.1) * np.ones(n_actions)
        )

        model = DDPG(
            "MlpPolicy",
            env,
            verbose=1,
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
                net_arch=dict(pi=actor_layers, qf=critic_layers)
            )
        )

    else:
        raise ValueError(f"Unsupported algo_type: {ALGO_TYPE}")

    print(f"{ALGO_TYPE} configured and initialized.")
else:
    if ALGO_TYPE == "PPO":
        model = PPO.load(model_path, env=env)

    elif ALGO_TYPE == "DDPG":
        model = DDPG.load(model_path, env=env)
    else:
        raise ValueError(f"Unsupported algo_type: {ALGO_TYPE}")
    print(f"Loaded existing {ALGO_TYPE} model.")

# === Training or evaluation ===
if not EVAL_MODE:
    model.learn(total_timesteps=int(PARAMS["total_episodes"] * PARAMS["episode_length"]))
    model.save(model_path)

    monitor_files = glob.glob(os.path.join(LOG_DIR, "*monitor.csv"))
    print("Monitor files found:", monitor_files)
    if not monitor_files:
        raise RuntimeError("No monitor CSV found. Check that total_timesteps >= n_steps")

    results = load_results(LOG_DIR)
    x, y = ts2xy(results, "timesteps")
    plot_results([results], LOG_DIR, "timesteps", model_name)
    plt.savefig(os.path.join(LOG_DIR, "reward_vs_steps.png"))
    print("Saved plot: reward_vs_steps.png")

else:
    for ep in range(PARAMS["total_episodes"]):
        obs = env.reset()
        total = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated = env.step(action)
            total += reward

# === Cleanup ===
sock_send.close()
sock_recv.close()
print("Training complete. Logs and plot saved in:", LOG_DIR)

