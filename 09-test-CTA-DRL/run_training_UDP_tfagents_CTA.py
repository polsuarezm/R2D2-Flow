import socket
import time
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
from tensorforce import Agent
from tensorforce.environments import Environment

# === Load Parameters ===
with open("input_parameters.json", "r") as f:
    PARAMS = json.load(f)

DEBUG = PARAMS.get("DEBUG", False)
LOG_DIR = PARAMS["log_dir_template"].format(datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(LOG_DIR, exist_ok=True)

# === Logging ===
log_buffer = deque(maxlen=10000)
def debug_log(msg, flush=False):
    if DEBUG:
        log_buffer.append(msg)
        if flush or len(log_buffer) >= 100:
            with open(os.path.join(LOG_DIR, "debug_log.txt"), "a") as f:
                f.write("\n".join(log_buffer) + "\n")
            log_buffer.clear()

# === UDP Setup ===
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind((PARAMS["hp_ip"], PARAMS["udp_port_recv"]))
print(f"Listening on {PARAMS['hp_ip']}:{PARAMS['udp_port_recv']}...")

# === Custom UDP Environment ===
class UdpEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.obs = None
        self.reward = None
        self.done = False
        self.timestamp = 0

    def states(self):
        return dict(type='float', shape=())

    def actions(self):
        return dict(type='float', min_value=20.0, max_value=1000.0)

    def reset(self):
        self.done = False
        self.obs, self.reward = self._get_obs_reward()
        return self.obs

    def execute(self, actions):
        # Send action to cRIO
        action_value = float(actions)
        action_int = int(action_value)
        hex_action = format(action_int, '016x')
        message = f"{hex_action};{self.timestamp}"
        sock_send.sendto(message.encode(), (PARAMS["crio_ip"], PARAMS["udp_port_send"]))

        # Get next obs and reward
        obs, reward = self._get_obs_reward()
        self.obs = obs
        self.reward = reward

        self.done = False  # No terminal signal unless implemented in future
        return obs, reward, self.done

    def _get_obs_reward(self):
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
                debug_log(f"[ERROR] Bad UDP format: {e}")
                continue

# === TensorForce Agent ===
environment = UdpEnvironment()
agent_config = {
    "agent": "ppo",
    "environment": environment,
    "batch_size": PARAMS.get("ppo_batch_size", 10),
    "learning_rate": PARAMS.get("ppo_learning_rate", 0.001),
    "update_frequency": PARAMS.get("ppo_update_frequency", 10),
    "max_episode_timesteps": PARAMS["episode_length"],
    "network": dict(type='auto', size=PARAMS["hidden_units"], depth=2),
    "summarizer": dict(directory=LOG_DIR, labels=["graph", "losses"]),
    "exploration": PARAMS.get("exploration", 0.1),
    "discount": 0.99
}

agent = Agent.create(**agent_config)

# === Evaluation Mode Option ===
EVAL_MODE = PARAMS.get("evaluation", False)
losses = []

if EVAL_MODE:
    print("🔍 Evaluation mode enabled — no learning will occur.")
    agent.load(directory=LOG_DIR, filename=None)

# === Main Loop ===
for episode in range(PARAMS["total_episodes"]):
    states = environment.reset()
    episode_reward = 0.0

    for t in range(PARAMS["episode_length"]):
        action = agent.act(states=states, independent=EVAL_MODE)
        states, reward, done = environment.execute(actions=action)
        if not EVAL_MODE:
            agent.observe(terminal=done, reward=reward)

        episode_reward += reward

        if DEBUG:
            debug_log(f"[Step {t}] Action: {action:.2f}, Reward: {reward:.2f}, State: {states:.2f}")

        if done:
            break

    print(f"[{'EVAL' if EVAL_MODE else 'TRAIN'}] Episode {episode + 1} reward: {episode_reward:.2f}")
    losses.append(episode_reward)

    if not EVAL_MODE and (episode + 1) % PARAMS.get("save_interval", 10) == 0:
        agent.save(directory=LOG_DIR, filename=f"ppo_episode_{episode + 1}")

# === Save final model or results ===
if not EVAL_MODE:
    agent.save(directory=LOG_DIR)

agent.close()
sock_send.close()
sock_recv.close()

def save_plot():
    plt.plot(losses)
    plt.title("Episode Reward Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(LOG_DIR, "reward_plot.png"), dpi=300)

save_plot()
print(f"{'Evaluation' if EVAL_MODE else 'Training'} complete. Plot saved.")
