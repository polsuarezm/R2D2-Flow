# === PPO TRAINING WITH KERAS ONLY (NO TFLITE) ===
import os
import socket
import time
import numpy as np
import tensorflow as tf
import json
from collections import deque
from datetime import datetime
from tensorflow.summary import create_file_writer

# === Load parameters ===
with open("input_parameters_tf.json", "r") as f:
    PARAMS = json.load(f)

PARAMS["log_dir"] = PARAMS["log_dir_template"].format(datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(PARAMS["log_dir"], exist_ok=True)
tensorboard_writer = create_file_writer(PARAMS["log_dir"])
DEBUG = PARAMS.get("DEBUG", False)

# === Logging ===
log_buffer = deque(maxlen=1000)
def debug_log(msg, flush=False):
    if DEBUG:
        log_buffer.append(msg)
        if flush or len(log_buffer) >= 100:
            with open(os.path.join(PARAMS["log_dir"], "debug_log.txt"), "a") as f:
                f.write("\n".join(log_buffer) + "\n")
            log_buffer.clear()

# === UDP Setup ===
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind((PARAMS["kv260_ip"], PARAMS["udp_port_recv"]))
print(f"Listening on {PARAMS['kv260_ip']}:{PARAMS['udp_port_recv']}...")

# === PPO Actor Model ===
def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(PARAMS["obs_dim"],)),
        tf.keras.layers.Dense(PARAMS["hidden1"], activation='relu'),
        tf.keras.layers.Dense(PARAMS["hidden2"], activation='relu'),
        tf.keras.layers.Dense(PARAMS["n_actions"], activation=None)
    ])

def extract_weights_as_string(model):
    weights = model.get_weights()
    flat_values = []
    for tensor in weights:
        flat_values.extend(tensor.flatten())

    arch_string = f"{PARAMS['obs_dim']}_{PARAMS['hidden1']}_{PARAMS['hidden2']}_{PARAMS['n_actions']}"
    identifier = "Control_id_x"
    comment = (
        "# Format: arch_id;"
        " W1_11;W1_12;...;W1_1n;W1_21;...;W1_mn;"
        " b1_1;...;b1_m;"
        " W2_11;...;W2_pq;"
        " b2_1;...;b2_p;"
        " W3_11;...;W3_rs;"
        " b3_1;...;b3_r;"
        " identifier\n"
    )
    final_string = arch_string + ";" + ";".join(f"{v:.5E}" for v in flat_values) + ";" + identifier
    total_params = len(flat_values)

    with open(PARAMS["weight_string_output"], "w") as f:
        f.write(comment)
        f.write(final_string)

    print("Export complete. Parameters:", total_params)
    print("Final string preview:", final_string[:200], "...")
    print("Total string length:", len(final_string))
    return final_string

# === PPO Loss Function ===
def ppo_loss(old_probs, advantages, actions, new_probs, clip_epsilon):
    prob_ratio = tf.exp(new_probs - old_probs)
    clipped_ratio = tf.clip_by_value(prob_ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    surrogate1 = prob_ratio * advantages
    surrogate2 = clipped_ratio * advantages
    return -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

# === Load/Create Model ===
if os.path.exists(PARAMS["model_path"]):
    model = tf.keras.models.load_model(PARAMS["model_path"])
else:
    model = create_model()
    model.save(PARAMS["model_path"])  # Save initially for Netron or inspection

optimizer = tf.keras.optimizers.Adam(learning_rate=PARAMS.get("learning_rate", 0.001))

# === Training Buffers ===
MAX_MEMORY = PARAMS["episode_length"]
obs_mem = np.zeros((MAX_MEMORY, PARAMS["obs_dim"]), dtype=np.float32)
act_mem = np.zeros((MAX_MEMORY, PARAMS["n_actions"]), dtype=np.float32)
rew_mem = np.zeros((MAX_MEMORY, 1), dtype=np.float32)

# === Initial Sync ===
weights_string = extract_weights_as_string(model)
sock_send.sendto(weights_string.encode(), (PARAMS["crio_ip"], PARAMS["udp_port_send"]))

# === Training Loop ===
episode = 0
step_idx = 0
while episode < PARAMS["total_episodes"]:
    for step in range(PARAMS["episode_length"]):
        data_rcv, _ = sock_recv.recvfrom(128)
        try:
            decoded = data_rcv.decode()
            st_str, at_str, rt_str = decoded.split(";")
            st = np.array([[float(x)] for x in st_str.split(",")], dtype=np.float32).T
            at = np.array([[float(x)] for x in at_str.split(",")], dtype=np.float32).T
            rt = np.array([[float(rt_str)]], dtype=np.float32)
        except Exception as e:
            debug_log(f"[Step {step}] Failed to parse UDP packet: {e}")
            continue

        obs_mem[step_idx] = st
        act_mem[step_idx] = at
        rew_mem[step_idx] = rt
        step_idx += 1

    if PARAMS["training"]:
        X = obs_mem[:step_idx]
        y = act_mem[:step_idx]
        rewards = rew_mem[:step_idx].flatten()
        advantages = rewards - np.mean(rewards)

        for _ in range(PARAMS.get("epochs_per_episode", 5)):
            with tf.GradientTape() as tape:
                logits = model(X, training=True)
                probs = tf.nn.sigmoid(logits)
                new_log_probs = tf.math.log(tf.clip_by_value(probs, 1e-10, 1.0)) * y + \
                                tf.math.log(tf.clip_by_value(1 - probs, 1e-10, 1.0)) * (1 - y)
                new_log_probs = tf.reduce_sum(new_log_probs, axis=1)
                old_log_probs = tf.stop_gradient(new_log_probs)
                adv_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
                loss = ppo_loss(old_log_probs, adv_tensor, y, new_log_probs, PARAMS.get("clip_epsilon", 0.2))

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        with tensorboard_writer.as_default():
            tf.summary.scalar("loss", loss, step=episode + 1)

        extract_weights_as_string(model)
        model.save(PARAMS["model_path"])
        print(f"[TRAIN] Episode {episode + 1} completed. Loss: {loss:.4f}")

        episode += 1
        step_idx = 0

# === Cleanup ===
sock_send.close()
sock_recv.close()
tensorboard_writer.close()
print("Training complete.")
