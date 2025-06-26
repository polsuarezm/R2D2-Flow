import socket, json, os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

class PlottingCallback(BaseCallback):
    """
    Logs metrics at each policy update and plots them live/after training.
    """
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.updates = []
        self.pg_losses = []
        self.clip_fracs = []
        self.ent_losses = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        idx = self.num_timesteps // self.model.n_steps
        def get(key):
            v = self.model.logger.name_to_value.get(key)
            return float(v) if v is not None else np.nan

        self.updates.append(idx)
        self.pg_losses.append(get("train/policy_gradient_loss"))
        self.clip_fracs.append(get("train/clip_fraction"))
        self.ent_losses.append(get("train/entropy_loss"))

        print(f"[Update {idx}] PG={self.pg_losses[-1]:.4f}, Clip={self.clip_fracs[-1]:.4f}, Ent={self.ent_losses[-1]:.4f}")

    def _on_training_end(self):
        plt.figure(figsize=(10,6))
        plt.plot(self.updates, self.pg_losses, label="Policy Gradient Loss")
        plt.plot(self.updates, self.clip_fracs, label="Clip Fraction")
        plt.plot(self.updates, self.ent_losses, label="Entropy Loss")
        plt.xlabel("Policy Update #")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.model.tensorboard_log or ".", "ppo_update_metrics.png"))
        plt.close()

# Load config
with open("input_parameters.json", "r") as f:
    PARAMS = json.load(f)

DEBUG = PARAMS.get("DEBUG", False)
EVAL_MODE = PARAMS.get("evaluation", False)
CREATE_NEW = PARAMS.get("create_new_model", True)
LOG_DIR = PARAMS["log_dir_template"].format(datetime.now().strftime("%Y%m%d-%H%M%S"))
LOG_DIR = PARAMS["log_dir_bueno"]
os.makedirs(LOG_DIR, exist_ok=True)

ACTION_MIN, ACTION_MAX = float(PARAMS["action_min"]), float(PARAMS["action_max"])
N_STEPS = int(PARAMS.get("n_steps", 2048))
BATCH_SIZE = int(PARAMS.get("batch_size", 64))
N_EPOCHS = int(PARAMS.get("n_epochs", 10))

# UDP setup
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind((PARAMS["hp_ip"], PARAMS["udp_port_recv"]))
print(f"Listening on {PARAMS['hp_ip']}:{PARAMS['udp_port_recv']}")

class CRIOUDPEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32)
        self.action_space = spaces.Box(ACTION_MIN, ACTION_MAX, (1,), dtype=np.float32)
        self.timestamp = 0
        self.step_count = 0
        self.last_obs = np.zeros(4, dtype=np.float32)

    def reset(self):
        self.step_count = 0
        return self._receive_obs()

    def step(self, action):
        raw = float(action[0])
        print(f"Raw action={raw:.4f}")

        output = raw*500+500
        sock_send.sendto(f"{self.timestamp};{output:.4f}".encode(),
                         (PARAMS["crio_ip"], PARAMS["udp_port_send"]))

        obs = self._receive_obs()
        reward = obs[1]-5
        print(f"step {self.step_count} -- reward {reward}")
        self.step_count += 1
        return obs, reward, self.step_count >= PARAMS["episode_length"], {}

    def _receive_obs(self):
        while True:
            data, _ = sock_recv.recvfrom(1024)
            parts = data.decode().strip().split(";")
            if len(parts) != 5:
                if DEBUG:
                    print(f"Bad packet: {data!r}")
                continue
            self.timestamp = int(parts[0])
            vals = np.array([float(x) for x in parts[1:5]], dtype=np.float32)
            self.last_obs = vals
            return vals

# Environment and model
env = DummyVecEnv([lambda: CRIOUDPEnv()])
model_path = os.path.join(LOG_DIR, "ppo_crio")

if CREATE_NEW or not os.path.exists(model_path + ".zip"):
    model = PPO(
        "MlpPolicy", env, verbose=1,
        learning_rate=PARAMS.get("ppo_learning_rate",1e-3),
        n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS,
        tensorboard_log=LOG_DIR,
        policy_kwargs=dict(net_arch=[PARAMS["hidden_units"]]*2),
    )
    print(f"PPO config: n_steps={N_STEPS}, batch_size={BATCH_SIZE}, n_epochs={N_EPOCHS}")
else:
    model = PPO.load(model_path, env=env)
    print("Loaded existing model.")

# Train or evaluate
if not EVAL_MODE:
    cb = PlottingCallback()
    model.learn(total_timesteps=int(PARAMS["total_episodes"]*PARAMS["episode_length"]),
                callback=cb, log_interval=1)
    model.save(model_path)
else:
    for ep in range(PARAMS["total_episodes"]):
        obs = env.reset()
        total = 0
        for _ in range(PARAMS["episode_length"]):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total += reward
        print(f"Evaluation Episode {ep+1}: Reward = {total:.2f}")

sock_send.close()
sock_recv.close()
print("Run complete. Plot saved to:", LOG_DIR)

