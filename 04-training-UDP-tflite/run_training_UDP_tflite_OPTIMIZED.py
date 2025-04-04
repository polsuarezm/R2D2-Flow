import socket
import time
import numpy as np
import tensorflow as tf
import struct
import os
import json
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.summary import create_file_writer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# === LOAD PARAMETERS FROM JSON ===
with open("input_parameters.json", "r") as f:
    PARAMS = json.load(f)

PARAMS["log_dir"] = PARAMS["log_dir_template"].format(datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(PARAMS["log_dir"], exist_ok=True)
tensorboard_writer = create_file_writer(PARAMS["log_dir"])
DEBUG = PARAMS.get("DEBUG", False)

# === Buffered Logging ===
log_buffer = deque(maxlen=1000) # Buffer size for log messages

def debug_log(msg, flush=False):  # Function to log debug messages
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

# === Model Functions ===
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)), # Input shape for single float
        tf.keras.layers.Dense(PARAMS["hidden_units"], activation='relu'), # First hidden layer
        tf.keras.layers.Dense(PARAMS["hidden_units"], activation='relu'), # Second hidden layer
        tf.keras.layers.Dense(PARAMS["n_actions"], activation='sigmoid') # Output layer (actions)
    ])
    return model

def save_weights_and_biases(model, step):
    weights = model.get_weights()
    for i, w in enumerate(weights):
        np.save(os.path.join(PARAMS["log_dir"], f"weights_step{step}_layer{i}.npy"), w)

def convert_to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(PARAMS["model_path"], 'wb') as f:
        f.write(tflite_model)

def load_or_create_model():
    if os.path.exists(PARAMS["model_path"]):
        interpreter = tf.lite.Interpreter(model_path=PARAMS["model_path"])
        interpreter.allocate_tensors()
        return interpreter, None
    else:
        model = create_model()
        convert_to_tflite(model)
        interpreter = tf.lite.Interpreter(model_path=PARAMS["model_path"])
        interpreter.allocate_tensors()
        return interpreter, model

interpreter, keras_model = load_or_create_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

print("TFLite input shape:", input_shape)

# === Training Setup ===
losses = []
episode_counter = 0
MAX_MEMORY = PARAMS["episode_length"]
obs_mem = np.zeros((MAX_MEMORY, 1), dtype=np.float32) # Memory for observations
act_mem = np.zeros((MAX_MEMORY, PARAMS["n_actions"]), dtype=np.uint8) # Memory for actions

if keras_model and PARAMS["training"]:
    keras_model.compile(optimizer='adam', loss='binary_crossentropy')

# === Main Loop ===
step_idx = 0
while episode_counter < PARAMS["total_episodes"]:
    for step in range(PARAMS["episode_length"]):
        t_start = time.time()
        data_rcv, _ = sock_recv.recvfrom(32)

        try:
            data_rcv_clean = data_rcv.decode()
            timestamp_str, obs_val_str, latency = data_rcv_clean.split(";")
            timestamp = int(timestamp_str)
            #print(f"laten >> {int(latency)}")
            obs_val = np.float32(obs_val_str)
            obs_array = np.array([[obs_val]], dtype=np.float32)
        except Exception as e:
            if DEBUG:
                debug_log(f"[Step {step}] Bad data format: {data_rcv_clean}")
            continue

        t_infer_start = time.time()
        interpreter.set_tensor(input_details[0]['index'], obs_array)
        interpreter.invoke()
        action = interpreter.get_tensor(output_details[0]['index']).flatten()
        t_infer_end = time.time()

        action_binary = (action >= 0.5).astype(np.uint8)
        action_str = ''.join(str(b) for b in action_binary)
        message = f"{format(int(action_str, 2), '016x')};{timestamp}"
        sock_send.sendto(message.encode(), (PARAMS["crio_ip"], PARAMS["udp_port_send"]))

        if PARAMS["training"] and keras_model:
            obs_mem[step_idx] = obs_val
            act_mem[step_idx] = action_binary
            step_idx += 1

        if DEBUG:
            total_time = (time.time() - t_start) * 1000
            infer_time = (t_infer_end - t_infer_start) * 1000
            debug_log(f"[TIMING] Step {step}: total={total_time:.2f} ms, inference={infer_time:.2f} ms")

    if PARAMS["training"] and keras_model:
        loss = keras_model.fit(obs_mem[:step_idx], act_mem[:step_idx], epochs=5, verbose=0).history['loss'][-1]
        losses.append(loss)
        convert_to_tflite(keras_model)
        save_weights_and_biases(keras_model, episode_counter + 1)

        with tensorboard_writer.as_default():
            tf.summary.scalar("loss", loss, step=episode_counter + 1)

        print(f"[TRAIN] Episode {episode_counter + 1} completed. Loss: {loss:.4f}")
        episode_counter += 1
        step_idx = 0

# === Final Plot ===
def save_plot():
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(PARAMS["log_dir"], "loss_curve.png"), dpi=500)

debug_log("Flushing remaining logs", flush=True)
sock_send.close()
sock_recv.close()
save_plot()
tensorboard_writer.close()
print("POOOL >> Training completed. Model saved and plot generated.")
