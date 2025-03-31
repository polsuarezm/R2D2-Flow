import socket
import time
import numpy as np
import tensorflow as tf
import struct
import os
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.summary import create_file_writer

# === CONFIG ===
TRAINING = True
N_ACT = 64
NUM_STEPS = 100000000000
EPISODE_LENGTH = 100  # You can tune this as needed
MODEL_PATH = "ppo_policy_dummy.tflite"
LOG_DIR = "logs/ppo_run_{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(LOG_DIR, exist_ok=True)
tensorboard_writer = create_file_writer(LOG_DIR)

# === UDP Setup ===
CRIO_IP = "172.22.10.2"
KV260_IP = "172.22.10.3"
UDP_PORT_SEND = 61557
UDP_PORT_RECV = 61555

sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind((KV260_IP, UDP_PORT_RECV))

print(f"Listening on {KV260_IP}:{UDP_PORT_RECV}...")

# === Model Setup ===
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

print("TFLite input shape:", input_shape)

# === Experience Buffer ===
episode_memory = []  # list of (obs, action)
episode_counter = 0
losses = []

def update_model(memory):
    # Placeholder for PPO or other RL algorithm
    # Here you'd convert the memory into training batches
    # and update the model weights
    dummy_loss = np.random.rand()  # Replace with actual loss
    losses.append(dummy_loss)
    print(f"[TRAIN] Updated model with {len(memory)} steps. Dummy loss: {dummy_loss:.4f}")
    return dummy_loss

def save_plot():
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(LOG_DIR, "loss_curve.png"))

# === Main Loop ===
for step in range(NUM_STEPS):
    data_rcv, _ = sock_recv.recvfrom(32)
    data_rcv_clean = data_rcv.decode()

    try:
        timestamp = int(data_rcv_clean.split(";")[0])
        obs_val = np.float32(data_rcv_clean.split(";")[1])
        obs_array = np.array([[obs_val]], dtype=np.float32)
    except:
        print(f"[Step {step}] Bad data format: {data_rcv_clean}")
        continue

    # === Inference ===
    interpreter.set_tensor(input_details[0]['index'], obs_array)
    interpreter.invoke()
    action = interpreter.get_tensor(output_details[0]['index']).flatten()
    action_binary = (action >= 0.5).astype(np.uint8)

    # === Action ===
    message = f"{format(int(action_str,2), '016x')};{timestamp}"
    #message = f"{format(action_binary, '016x')};{timestamp}"
    sock_send.sendto(message.encode(), (CRIO_IP, UDP_PORT_SEND))

    # === Store for training ===
    if TRAINING:
        episode_memory.append((obs_array.copy(), action_binary.copy()))

        if len(episode_memory) >= EPISODE_LENGTH:
            episode_counter += 1
            loss = update_model(episode_memory)
            episode_memory = []

            # === Log to TensorBoard ===
            with tensorboard_writer.as_default():
                tf.summary.scalar("loss", loss, step=episode_counter)
            print(f"[TRAIN] Episode {episode_counter} completed.")

            # Optional: save updated model weights here (re-export tflite)

    # === Optional sleep for pacing ===
    # time.sleep(0.01)

# === Cleanup ===
sock_send.close()
sock_recv.close()
save_plot()
