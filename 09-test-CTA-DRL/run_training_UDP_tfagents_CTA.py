import os
import json
import socket
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from datetime import datetime
from collections import deque

from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.utils import common
from tf_agents.drivers import dynamic_step_driver
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory

# === Load Params ===
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

# === UDP Setup ===
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind((PARAMS["pc_ip"], PARAMS["udp_port_recv"]))
print(f"Listening on {PARAMS['pc_ip']}:{PARAMS['udp_port_recv']}...")

# === UDP Environment for TF-Agents ===
class UdpEnv(py_environment.PyEnvironment):
    def __init__(self):
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.float32, minimum=-np.inf, maximum=np.inf, name='observation'
        )
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.float32, minimum=20.0, maximum=1000.0, name='action'
        )
        self._state = 0.0
        self._reward = 0.0
        self._done = False
        self._timestamp = 0
        self._step_count = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._step_count = 0
        self._state, self._reward = self._recv_udp()
        return ts.restart(np.array(self._state, dtype=np.float32))

    def _step(self, action):
        if self._done:
            return self.reset()

        action_int = int(action)
        action_hex = format(action_int, '016x')
        message = f"{action_hex};{self._timestamp}"
        sock_send.sendto(message.encode(), (PARAMS["crio_ip"], PARAMS["udp_port_send"]))

        self._state, self._reward = self._recv_udp()
        self._step_count += 1
        if self._step_count >= PARAMS["episode_length"]:
            self._done = True
            return ts.termination(np.array(self._state, dtype=np.float32), self._reward)
        else:
            return ts.transition(np.array(self._state, dtype=np.float32), reward=self._reward)

    def _recv_udp(self):
        while True:
            try:
                data_rcv, _ = sock_recv.recvfrom(32)
                decoded = data_rcv.decode()
                parts = decoded.strip().split(";")
                if len(parts) < 3:
                    continue
                self._timestamp = int(parts[0])
                obs = float(parts[1])
                rew = float(parts[2])
                return obs, rew
            except Exception as e:
                debug_log(f"UDP error: {e}")
                continue

# === Create TF Environments ===
env = UdpEnv()
tf_env = tf_py_environment.TFPyEnvironment(env)

# === Create Networks ===
actor_net = ActorDistributionNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    fc_layer_params=(PARAMS["hidden_units"],)
)
value_net = ValueNetwork(
    tf_env.observation_spec(),
    fc_layer_params=(PARAMS["hidden_units"],)
)

# === Optimizer and Agent ===
global_step = tf.compat.v1.train.get_or_create_global_step()
optimizer = tf.keras.optimizers.Adam(learning_rate=PARAMS.get("ppo_learning_rate", 1e-3))

tf_agent = ppo_agent.PPOAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    actor_net=actor_net,
    value_net=value_net,
    optimizer=optimizer,
    normalize_rewards=True,
    train_step_counter=global_step
)
tf_agent.initialize()

# === Policy and Replay Buffer ===
eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=1000
)

# === Data Collection Function ===
def collect_step(environment, policy):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    replay_buffer.add_batch(traj)

# === Evaluation or Training ===
if EVAL_MODE:
    print("🔍 Evaluation Mode")
    policy_saver.PolicySaver(eval_policy).save(LOG_DIR)
    for episode in range(PARAMS["total_episodes"]):
        time_step = tf_env.reset()
        total_reward = 0.0
        for _ in range(PARAMS["episode_length"]):
            action_step = eval_policy.action(time_step)
            time_step = tf_env.step(action_step.action)
            total_reward += time_step.reward.numpy()[0]
        print(f"[EVAL] Episode {episode+1} reward: {total_reward:.2f}")
else:
    print("🚀 Training Mode")
    tf_agent.train = common.function(tf_agent.train)
    returns = []

    for episode in range(PARAMS["total_episodes"]):
        tf_env.reset()
        for _ in range(PARAMS["episode_length"]):
            collect_step(tf_env, collect_policy)

        experience = replay_buffer.gather_all()
        train_loss = tf_agent.train(experience)
        replay_buffer.clear()

        episode_return = float(train_loss.loss.numpy())
        returns.append(episode_return)
        print(f"[TRAIN] Episode {episode+1}, Loss: {episode_return:.4f}")

    # Save model and plot
    policy_dir = os.path.join(LOG_DIR, "policy")
    policy_saver.PolicySaver(tf_agent.policy).save(policy_dir)

    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(os.path.join(LOG_DIR, "training_loss.png"))

# === Cleanup ===
sock_send.close()
sock_recv.close()

