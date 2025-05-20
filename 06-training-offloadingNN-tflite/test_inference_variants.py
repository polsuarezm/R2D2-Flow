import numpy as np
import tensorflow as tf
import json
import os

# === Define 3 test parameter sets ===
PARAM_LIST = [
    { "obs_dim": 3, "hidden1": 4, "hidden2": 6, "n_actions": 2 },
    { "obs_dim": 2, "hidden1": 3, "hidden2": 4, "n_actions": 5 },
    { "obs_dim": 3, "hidden1": 4, "hidden2": 5, "n_actions": 6 }
]

OUTPUT_DIR = "inference_debug"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Model creation using PARAMS
def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(PARAMS["obs_dim"],)),
        tf.keras.layers.Dense(PARAMS["hidden1"], activation='relu'),
        tf.keras.layers.Dense(PARAMS["hidden2"], activation='relu'),
        tf.keras.layers.Dense(PARAMS["n_actions"], activation=None)
    ])

# === Export weights string like UDP format
def export_weights_string(model, PARAMS):
    weights = model.get_weights()
    flat_values = [v for tensor in weights for v in tensor.flatten()]
    arch_string = f"{PARAMS['obs_dim']}_{PARAMS['hidden1']}_{PARAMS['hidden2']}_{PARAMS['n_actions']}"
    final_string = arch_string + ";" + ";".join(f"{v:.5E}" for v in flat_values) + ";Control_id_x"
    return final_string

# === Save helper
def save_txt(filename, string):
    with open(filename, "w") as f:
        f.write(string)

# === Run tests
for i, PARAMS in enumerate(PARAM_LIST, start=1):
    print(f"\n=== ARCH {i}: {PARAMS} ===")

    # Build model from PARAMS
    model = create_model()

    # Save weights string
    weight_string = export_weights_string(model, PARAMS)
    weights_file = os.path.join(OUTPUT_DIR, f"weights_arch{i}.txt")
    save_txt(weights_file, weight_string)
    print(f"Saved weights string to {weights_file}")

    # Run 3 inferences
    actions = []
    actions_text = ""
    for j in range(3):
        fake_obs = np.random.rand(1, PARAMS["obs_dim"]).astype(np.float32)
        logits = model(fake_obs)
        probs = tf.nn.softmax(logits).numpy().flatten()
        actions.append(probs.tolist())
        actions_text += ";".join(f"{p:.8f}" for p in probs) + "\n"
        print(f"  [Input {j+1}] obs: {fake_obs.flatten()} → probs: {probs}")

    # Save output (UDP-style)
    actions_file = os.path.join(OUTPUT_DIR, f"actions_arch{i}.txt")
    save_txt(actions_file, actions_text.strip())
    print(f"Saved action outputs to {actions_file}")

    # Save output (NumPy array-style)
    actions_array_file = os.path.join(OUTPUT_DIR, f"actions_arch{i}_array.txt")
    np.savetxt(actions_array_file, np.array(actions), fmt="%.8f", delimiter=" ")
    print(f"Saved actions as array to {actions_array_file}")
