import numpy as np
import tensorflow as tf
import os

ARCH_LIST = [
    { "obs_dim": 3, "hidden1": 4, "hidden2": 6, "n_actions": 2 },
    { "obs_dim": 2, "hidden1": 3, "hidden2": 4, "n_actions": 5 },
    { "obs_dim": 3, "hidden1": 4, "hidden2": 5, "n_actions": 6 }
]

def create_model(PARAMS):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(PARAMS["obs_dim"],)),
        tf.keras.layers.Dense(PARAMS["hidden1"], activation='relu'),
        tf.keras.layers.Dense(PARAMS["hidden2"], activation='relu'),
        tf.keras.layers.Dense(PARAMS["n_actions"], activation=None)
    ])

def export_weights_string(model, PARAMS):
    weights = model.get_weights()
    flat_values = [v for tensor in weights for v in tensor.flatten()]
    arch_string = f"{PARAMS['obs_dim']}_{PARAMS['hidden1']}_{PARAMS['hidden2']}_{PARAMS['n_actions']}"
    return arch_string + ";" + ";".join(f"{v:.5E}" for v in flat_values) + ";Control_id_x"

def save_udp_string(values, filepath):
    with open(filepath, "w") as f:
        f.write(";".join(f"{v:.8f}" for v in values))

for i, PARAMS in enumerate(ARCH_LIST, start=1):
    print(f"\n=== ARCH {i}: {PARAMS} ===")

    model = create_model(PARAMS)

    # Save weights
    weights_string = export_weights_string(model, PARAMS)
    with open(f"weights_arch{i}.txt", "w") as f:
        f.write(weights_string)
    print(f"Saved weights_arch{i}.txt")

    # Generate 1 input observation
    obs = np.random.rand(PARAMS["obs_dim"]).astype(np.float32)

    # Save observation
    save_udp_string(obs, f"obs_arch{i}.txt")
    print(f"Saved obs_arch{i}.txt")

    # Run model and get softmax output
    logits = model(obs.reshape(1, -1))
    probs = tf.nn.softmax(logits).numpy().flatten()

    # Save action output
    save_udp_string(probs, f"actions_arch{i}.txt")
    print(f"Saved actions_arch{i}.txt")
