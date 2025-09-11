import socket, json, os, glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results

# === Load configuration ===
with open("input_parameters_v2.json", "r") as f:
    PARAMS = json.load(f)

DEBUG = PARAMS.get("DEBUG", False)
EVAL_MODE = PARAMS.get("evaluation", False)
CREATE_NEW = PARAMS.get("create_new_model", True)

LOG_DIR = PARAMS["log_dir_template"].format(datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(LOG_DIR, exist_ok=True)

ACTION_MIN     = float(PARAMS["action_min"])
ACTION_MAX     = float(PARAMS["action_max"])
N_STEPS        = int(PARAMS["n_steps"])
BATCH_SIZE     = int(PARAMS["batch_size"])
N_EPOCHS       = int(PARAMS["n_epochs"])
N_OBS_ARRAY    = int(PARAMS["size_obs_array"])
N_ACTUATOR_ARRAY = int(PARAMS["size_actuator_array"])
MESSAGE_TYPE   = int(PARAMS["message_type"])  # 1 for array, 2 for string
SCALAR_REW     = float(PARAMS["scalar_reward"])
TOTAL_DESCARTE = int(PARAMS["total_descarte"]) 
# === UDP setup ===
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.setblocking(False)
sock_recv.bind((PARAMS["hp_ip"], PARAMS["udp_port_recv"]))
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
        self.last_obs = np.zeros(N_OBS_ARRAY, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        obs, aux_obs = self._receive_observation()
        return obs, {}

    def step(self, action):
        
        activate_mask = (action[0:6] > 0).astype(int)
        freq_mask     = (action[6]+1)/2

        message = f"{self.timestamp};" + ';'.join(map(str, activate_mask)) + f";{freq_mask}"

        sock_send.sendto(
            message.encode(),
            (PARAMS["crio_ip"], PARAMS["udp_port_send"])
        )

        obs, aux_obs = self._receive_observation()

        reward = (SCALAR_REW - obs[2]) / SCALAR_REW
        # Log reward to file
        with open(os.path.join(LOG_DIR, "live_rewards.csv"), "a") as f:
            f.write(f"{self.step_count},{reward},{self.timestamp}\n")

        self.step_count += 1
        terminated = self.step_count >= PARAMS["episode_length"]
        truncated = False

        if DEBUG:
            print("Received obs:", reward, "action used:", aux_obs, "ts:", self.timestamp)


        #if DEBUG:
         #   print(f"{self.step_count};{raw_action};{obs[2]:.6f};{reward:.4f}")

        return obs, reward, terminated, truncated, {}

    def _receive_observation(self):
        # Vaciar el buffer UDP
        sock_recv.setblocking(False)
        while True:
            try:
                sock_recv.recvfrom(1024)
            except BlockingIOError:
                break

        # Recibir siguiente observación

        sock_recv.setblocking(True)

        for i_descarte in range(TOTAL_DESCARTE + 1): 
            sock_recv.setblocking(True)
            data, _ = sock_recv.recvfrom(1024)
            parts = data.decode().strip().split(";")
            aux_obs = np.array([float(x) for x in parts[1:6]], dtype=np.float32)        
            print(f"descarte: rew = {(SCALAR_REW - aux_obs[2]) / SCALAR_REW}; action = {aux_obs[4]})")
        
        sock_recv.setblocking(False)

        self.timestamp = int(parts[0])
        self.last_obs = np.array([float(x) for x in parts[1:5]], dtype=np.float32)


        return self.last_obs, aux_obs[4]

    def render(self, mode="human"): pass
    def close(self): pass

# === Wrap environment with Monitor ===
def make_env():
    env = CRIOUDPEnv()
    return Monitor(env, filename=os.path.join(LOG_DIR, "env_monitor"))

env = DummyVecEnv([make_env])

# === PPO model setup ===
model_path = os.path.join(LOG_DIR, "ppo_crio")
if CREATE_NEW or not os.path.exists(model_path + ".zip"):
    model = PPO(
        "MlpPolicy", env, verbose=1,
        learning_rate=PARAMS.get("ppo_learning_rate", 1e-3),
        n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS,
        tensorboard_log=LOG_DIR,
        policy_kwargs=dict(net_arch=[PARAMS["hidden_units"]] * 2),
    )
    print(f"PPO configured: n_steps={N_STEPS}, batch_size={BATCH_SIZE}, n_epochs={N_EPOCHS}")
else:
    model = PPO.load(model_path, env=env)
    print("Loaded existing model.")

# === Training or evaluation ===
if not EVAL_MODE:
    model.learn(total_timesteps=int(PARAMS["total_episodes"] * PARAMS["episode_length"]))
    model.save(model_path)

    monitor_files = glob.glob(os.path.join(LOG_DIR, "*monitor.csv"))
    print("Monitor files found:", monitor_files)
    if not monitor_files:
        raise RuntimeError("No monitor CSV found. "
                           "Check that total_timesteps >= n_steps to allow PPO updates.")

    results = load_results(LOG_DIR)
    x, y = ts2xy(results, "timesteps")
    plot_results([results], LOG_DIR, "timesteps", "ppo_crio")
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
        print(f"Eval Episode {ep+1}: Total Reward = {total:.2f}")

# === Cleanup ===
sock_send.close()
sock_recv.close()
print("Training complete. Logs and plot saved in:", LOG_DIR)

