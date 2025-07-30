#!/usr/bin/env python3
"""
UDP‑based PPO Training Loop with Keras (no TFLite)
Receives full episode trajectories via UDP, trains a PPO policy,
and sends updated weights back to CRIO via UDP.
"""

import os
import socket
import time
import numpy as np
import tensorflow as tf
import json
from datetime import datetime
from tensorflow.summary import create_file_writer

# ── 1. Configuration & Logging Setup ───────────────────────────────
with open("input_parameters_tf.json", "r") as f:
    PARAMS = json.load(f)

# Log directory setup
PARAMS["log_dir"] = PARAMS["log_dir_template"].format(datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(PARAMS["log_dir"], exist_ok=True)
tensorboard_writer = create_file_writer(PARAMS["log_dir"])
DEBUG = PARAMS.get("DEBUG", False)

# ── 2. UDP Socket Initialization ─────────────────────────────────
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

sock_recv.bind(("0.0.0.0", PARAMS["udp_port_recv"]))
sock_recv.settimeout(5)  # timeout for trajectory reception

target_address = (PARAMS["crio_ip"], PARAMS["udp_port_send"])

# ── 3. Model Definition & Weight Serialization ────────────────────
def create_model():
    """Create the Keras actor network for PPO."""
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(PARAMS["obs_dim"],)),
        tf.keras.layers.Dense(PARAMS["hidden1"], activation='relu'),
        tf.keras.layers.Dense(PARAMS["hidden2"], activation='relu'),
        tf.keras.layers.Dense(PARAMS["n_actions"], activation=None)
    ])

def extract_weights_as_string(model):
    """
    Flatten weights into a single string:
    arch_header;weight1;...;weightN;identifier
    Saves model string to disk too.
    """
    weights = model.get_weights()
    flat_vals = [v for w in weights for v in w.flatten()]
    arch_string = f"{PARAMS['obs_dim']}_{PARAMS['hidden1']}_{PARAMS['hidden2']}_{PARAMS['n_actions']}"
    identifier = PARAMS.get("identifier_str", "Control_id_x")
    header = "# arch; weights; identifier\n"
    body = arch_string + ";" + ";".join(f"{v:.5E}" for v in flat_vals) + ";" + identifier
    with open(PARAMS["weight_string_output"], "w") as f:
        f.write(header + body)
    return body

# ── 4. PPO Loss Definition ─────────────────────────────────────────
def ppo_loss(old_log_probs, advantages, actions, new_log_probs, clip_epsilon):
    """
    Standard PPO clipped surrogate objective.
    old_log_probs: from previous policy (stop-gradient).
    new_log_probs: re-computed from current logits.
    advantages: advantage estimates.
    """
    ratio = tf.exp(new_log_probs - old_log_probs)
    clipped = tf.clip_by_value(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    return -tf.reduce_mean(tf.minimum(ratio * advantages, clipped * advantages))

# ── 5. Model Load / Initialization ─────────────────────────────────
if os.path.exists(PARAMS["model_path"]):
    model = tf.keras.models.load_model(PARAMS["model_path"])
else:
    model = create_model()
    model.save(PARAMS["model_path"])

optimizer = tf.keras.optimizers.Adam(learning_rate=PARAMS.get("learning_rate", 1e-3))

# ── 6. Trajectory Buffers for One Episode ─────────────────────────
MAX_MEMORY = PARAMS["episode_length"]
obs_mem = np.zeros((MAX_MEMORY, PARAMS["obs_dim"]), dtype=np.float32)
act_mem = np.zeros((MAX_MEMORY, PARAMS["n_actions"]), dtype=np.float32)
rew_mem = np.zeros((MAX_MEMORY, 1), dtype=np.float32)

# ── 7. Initial Model Sync (on startup) ────────────────────────────
initial_weights = extract_weights_as_string(model)
sock_send.sendto(initial_weights.encode(), target_address)
print("[INIT] Sent initial model weights to CRIO.")

# ── 8. Main Training Loop Over Episodes ───────────────────────────
episode = 0
while episode < PARAMS["total_episodes"]:
    print(f"[UDP] Awaiting data for episode {episode + 1}")
    step_idx = 0

    # ● Receive one episode trajectory
    while step_idx < MAX_MEMORY:
        try:
            data, _ = sock_recv.recvfrom(2048)
            if data == b"<END>":
                break
            decoded = data.decode()
            st_str, at_str, rt_str = decoded.split(";")
            st = np.array([[float(x)] for x in st_str.split(",")], dtype=np.float32).T
            at = np.array([[float(x)] for x in at_str.split(",")], dtype=np.float32).T
            rt = np.array([[float(rt_str)]], dtype=np.float32)

            obs_mem[step_idx] = st
            act_mem[step_idx] = at
            rew_mem[step_idx] = rt
            step_idx += 1
        except Exception as e:
            print(f"[WARN] Failed to parse packet: {e}")

    print(f"[INFO] Collected {step_idx} steps; starting training.")

    # ● Perform PPO training update
    if PARAMS.get("training", True):
        X = obs_mem[:step_idx]
        y = act_mem[:step_idx]
        rewards = rew_mem[:step_idx].flatten()
        advantages = rewards - np.mean(rewards)

        for _ in range(PARAMS.get("epochs_per_episode", 5)):
            with tf.GradientTape() as tape:
                logits = model(X, training=True)
                probs = tf.nn.sigmoid(logits)
                new_log = tf.reduce_sum(
                    tf.math.log(tf.clip_by_value(probs, 1e-10, 1.0)) * y +
                    tf.math.log(tf.clip_by_value(1 - probs, 1e-10, 1.0)) * (1 - y),
                    axis=1
                )
                old_log = tf.stop_gradient(new_log)
                loss = ppo_loss(old_log, tf.convert_to_tensor(advantages, tf.float32), y,
                                new_log, PARAMS.get("clip_epsilon", 0.2))

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # ● Save model and send updated weights via UDP
        model.save(PARAMS["model_path"])
        new_weights = extract_weights_as_string(model)
        sock_send.sendto(new_weights.encode(), target_address)
        print(f"[TRAIN] Episode {episode + 1} done; Loss: {loss.numpy():.4f}")

        with tensorboard_writer.as_default():
            tf.summary.scalar("loss", loss, step=episode + 1)

    episode += 1

# ── 9. Final Cleanup ────────────────────────────────────────────────
sock_send.close()
sock_recv.close()
tensorboard_writer.close()
print("✅ Training session finished.")
