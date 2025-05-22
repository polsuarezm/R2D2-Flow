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
        tf.keras.layers.Dense(PARAMS["n_actions"], activation=None)  # No activation
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
    with open(f"fix_weights_arch{i}.txt", "w") as f:
        f.write(weights_string)
    print(f"Saved fix_weights_arch{i}.txt")

    # Generate random observation
    obs = np.random.rand(PARAMS["obs_dim"]).astype(np.float32)
    save_udp_string(obs, f"fix_obs_arch{i}.txt")
    print(f"Saved fix_obs_arch{i}.txt")

    # Manual forward pass
    weights = model.get_weights()
    W1, b1 = weights[0], weights[1]
    W2, b2 = weights[2], weights[3]
    W3, b3 = weights[4], weights[5]

    z1 = np.maximum(0, np.dot(obs, W1) + b1)
    save_udp_string(z1, f"fix_layer1_arch{i}.txt")
    print(f"Saved fix_layer1_arch{i}.txt")

    z2 = np.maximum(0, np.dot(z1, W2) + b2)
    save_udp_string(z2, f"fix_layer2_arch{i}.txt")
    print(f"Saved fix_layer2_arch{i}.txt")

    logits = np.dot(z2, W3) + b3
    output = logits.flatten()
    save_udp_string(output, f"fix_actions_arch{i}.txt")
    print(f"Saved fix_actions_arch{i}.txt (raw logits)")

    # Compare with model() output
    logits_tf = model(obs.reshape(1, -1)).numpy().flatten()
    print("Manual logits:     ", output)
    print("TF model logits:   ", logits_tf)
    print("Abs. difference:   ", np.abs(output - logits_tf))

    assert np.allclose(output, logits_tf, atol=1e-6), "Mismatch between manual and TF logits!"
