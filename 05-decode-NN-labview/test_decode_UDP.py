import os
import numpy as np
import tensorflow as tf

# === Configurable architecture ===
obs_dim = 2
hidden1 = 3
hidden2 = 4
action_dim = 5

MODEL_PATH = "ppo_actor.tflite"
OUTPUT_PATH = "weights_encoded_with_architecture.txt"
IDENTIFIER = "Control_id_x"

# === Step 1: Build PPO Actor Model ===
def create_actor():
    inputs = tf.keras.Input(shape=(obs_dim,), name="observation")
    x = tf.keras.layers.Dense(hidden1, activation='relu', name="dense_1")(inputs)
    x = tf.keras.layers.Dense(hidden2, activation='relu', name="dense_2")(x)
    outputs = tf.keras.layers.Dense(action_dim, activation=None, name="action")(x)
    return tf.keras.Model(inputs, outputs)

model = create_actor()
model.save_weights("ppo_actor.weights.h5")

# === Step 2: Convert to TFLite ===
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(MODEL_PATH, "wb") as f:
    f.write(tflite_model)
print(f"\nTFLite model saved to {MODEL_PATH}")

# === Step 3: Load TFLite model and extract tensors ===
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, experimental_preserve_all_tensors=True)
interpreter.allocate_tensors()
details = interpreter.get_tensor_details()

# === Collect all constant tensors ===
all_constants = []
for d in details:
    name = d['name']
    try:
        tensor = interpreter.tensor(d['index'])()
    except ValueError:
        continue
    all_constants.append((name, tensor))

# === Filter valid weights and biases ===
candidate_weights = [
    (name, t) for name, t in all_constants if t.ndim == 2 and t.shape[0] > 1
]
candidate_biases = [
    (name, t.reshape(-1)) for name, t in all_constants
    if (t.ndim == 1 or (t.ndim == 2 and (t.shape[0] == 1 or t.shape[1] == 1)))
]

# Sort to get correct layer order
candidate_weights.sort(key=lambda x: x[1].shape[1])
candidate_biases.sort(key=lambda x: x[1].shape[0])

# === Safety check ===
if len(candidate_weights) < 3 or len(candidate_biases) < 3:
    print("Error: Not enough weights or biases found.")
    exit(1)

# === Assign weights and biases ===
W1 = candidate_weights[0][1]
b1 = candidate_biases[0][1]
W2 = candidate_weights[1][1]
b2 = candidate_biases[1][1]
W3 = candidate_weights[2][1]
b3 = candidate_biases[2][1]

# === Architecture from weights ===
input_size = W1.shape[1]
hidden1_size = W1.shape[0]
hidden2_size = W2.shape[0]
output_size = W3.shape[0]

print(f"\nArchitecture: {input_size}_{hidden1_size}_{hidden2_size}_{output_size}")
print("Layer shapes:")
print(f"  W1: {W1.shape}, b1: {b1.shape}")
print(f"  W2: {W2.shape}, b2: {b2.shape}")
print(f"  W3: {W3.shape}, b3: {b3.shape}")

# === Flatten and export ===
tensors_in_order = [W1, b1, W2, b2, W3, b3]
flat_values = []
for tensor in tensors_in_order:
    flat_values.extend(tensor.flatten())

total_params = len(flat_values)

# === Compose final string ===
arch_string = f"{input_size}_{hidden1_size}_{hidden2_size}_{output_size}"
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

final_string = arch_string + ";" + ";".join(f"{v:.8f}" for v in flat_values) + ";" + IDENTIFIER

# === Save to file ===
with open(OUTPUT_PATH, "w") as f:
    f.write(comment)
    f.write(final_string)

# === Final output ===
print(f"\nExport complete. Saved to {OUTPUT_PATH}")
print("Final architecture string:", arch_string)
print("Number of trainable parameters:", total_params)
print("Encoded string preview:")
print(final_string)
print("Total string length:", len(final_string))
