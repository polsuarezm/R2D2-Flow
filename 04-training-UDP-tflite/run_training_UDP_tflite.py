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

# === LOAD PARAMETERS FROM JSON ===
with open("config.json", "r") as f:
    PARAMS = json.load(f)

PARAMS["log_dir"] = PARAMS["log_dir_template"].format(datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(PARAMS["log_dir"], exist_ok=True)
tensorboard_writer = create_file_writer(PARAMS["log_dir"])
DEBUG = PARAMS.get("DEBUG", False)

def debug_log(msg):
    if DEBUG:
        with open(os.path.join(PARAMS["log_dir"], "debug_log.txt"), "a") as f:
            f.write(msg + "\n")

# === UDP Setup (specific for Carlos CRIO+KV260 in UPV) ===
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind((PARAMS["kv260_ip"], PARAMS["udp_port_recv"]))

print(f"Listening on {PARAMS['kv260_ip']}:{PARAMS['udp_port_recv']}...")

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(PARAMS["hidden_units"], activation='relu'),
        tf.keras.layers.Dense(PARAMS["hidden_units"], activation='relu'),
        tf.keras.layers.Dense(PARAMS["n_actions"], activation='sigmoid')
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

# === Experience Buffer ===
episode_memory = []
episode_counter = 0
losses = []

def update_model(memory, model):
    start = time.time()
    obs_batch = np.array([m[0][0] for m in memory])
    act_batch = np.array([m[1] for m in memory])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    history = model.fit(obs_batch, act_batch, epochs=5, verbose=0)
    loss = history.history['loss'][-1]
    convert_to_tflite(model)
    losses.append(loss)
    duration = (time.time() - start) * 1000
    debug_log(f"[TIMING] Model update duration: {duration:.2f} ms")
    print(f"[TRAIN] Updated model with {len(memory)} steps. Loss: {loss:.4f}")
    return loss

def save_plot():
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(PARAMS["log_dir"], "loss_curve.png"), dpi=500)

# === Main Loop ===
while episode_counter < PARAMS["total_episodes"]:
    for step in range(PARAMS["episode_length"]):
        t_start = time.time()

        data_rcv, _ = sock_recv.recvfrom(32)
        data_rcv_clean = data_rcv.decode()

        try:
            timestamp = int(data_rcv_clean.split(";")[0])
            obs_val = np.float32(data_rcv_clean.split(";")[1])
            obs_array = np.array([[obs_val]], dtype=np.float32)
        except:
            print(f"[Step {step}] Bad data format: {data_rcv_clean}")
            continue

        t_infer_start = time.time()
        interpreter.set_tensor(input_details[0]['index'], obs_array)
        interpreter.invoke()
        action = interpreter.get_tensor(output_details[0]['index']).flatten()
        t_infer_end = time.time()

        action_binary = (action >= 0.5).astype(np.uint8)
        action_str = ''.join(str(b) for b in action_binary)
        message = f"{format(int(action_str,2), '016x')};{timestamp}"
        sock_send.sendto(message.encode(), (PARAMS["crio_ip"], PARAMS["udp_port_send"]))

        if PARAMS["training"] and keras_model:
            episode_memory.append((obs_array.copy(), action_binary.copy()))

        t_end = time.time()
        if DEBUG:
            debug_log(f"[TIMING] Step {step}: total={((t_end - t_start) * 1000):.2f} ms, inference={((t_infer_end - t_infer_start) * 1000):.2f} ms")

    if PARAMS["training"] and keras_model:
        episode_counter += 1
        loss = update_model(episode_memory, keras_model)
        save_weights_and_biases(keras_model, episode_counter)
        episode_memory = []

        with tensorboard_writer.as_default():
            tf.summary.scalar("loss", loss, step=episode_counter)
        print(f"[TRAIN] Episode {episode_counter} completed.")

# === Cleanup ===
sock_send.close()
sock_recv.close()
save_plot()
tensorboard_writer.close()
print("POOOL >> Training completed. Model saved and plot generated.")