import os
import numpy as np
import tensorflow as tf

MODEL_PATH = "ppo_policy_todecode.tflite"
OUTPUT_PATH = "weights_encoded_with_architecture.txt"

# Load the model
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file not found: {MODEL_PATH}")
    exit(1)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, experimental_preserve_all_tensors=True)
interpreter.allocate_tensors()

details = interpreter.get_tensor_details()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Get architecture sizes ===
input_size = input_details[0]['shape'][1]
output_size = output_details[0]['shape'][1]

candidate_weights = []
candidate_biases = []

print("\n=== Searching for candidate weights and biases ===")

for d in details:
    tensor = interpreter.tensor(d['index'])()
    name = d['name']
    shape = tensor.shape
    print(f"Name: {name}, Shape: {shape}")

    if tensor.ndim == 2:
        candidate_weights.append((name, tensor))
    elif tensor.ndim == 1:
        candidate_biases.append((name, tensor))

# Sort by output size (optional but useful)
candidate_weights.sort(key=lambda x: x[1].shape[1])
candidate_biases.sort(key=lambda x: x[1].shape[0])

# === Sanity check ===
if len(candidate_weights) < 3 or len(candidate_biases) < 3:
    print("\n❌ Not enough weight or bias tensors found!")
    print(f"   Weights found: {len(candidate_weights)}")
    print(f"   Biases found: {len(candidate_biases)}")
    exit(1)

# === Assign in proper order ===
W1 = candidate_weights[0][1]
b1 = candidate_biases[0][1]
W2 = candidate_weights[1][1]
b2 = candidate_biases[1][1]
W3 = candidate_weights[2][1]
b3 = candidate_biases[2][1]

hidden1_size = W1.shape[1]
hidden2_size = W2.shape[1]

# Confirm layer sizes
print(f"\nArchitecture: {input_size}_{hidden1_size}_{hidden2_size}_{output_size}")

# === Flatten weights in order ===
tensors_in_order = [W1, b1, W2, b2, W3, b3]
flat_values = []

for i, tensor in enumerate(tensors_in_order):
    flat = tensor.flatten()
    flat_values.extend(flat)
    print(f"Tensor {i+1}: shape={tensor.shape}, values={flat.size}")

# === Total count
total_count = len(flat_values)
print(f"\nTotal parameters encoded: {total_count}")

# === Format string
arch_string = f"{input_size}_{hidden1_size}_{hidden2_size}_{output_size}"
final_string = arch_string + ";" + ";".join(f"{v:.8f}" for v in flat_values)

# === Save to file
with open(OUTPUT_PATH, "w") as f:
    f.write(final_string)

print(f"\nExport complete. Saved to {OUTPUT_PATH}")
print("String preview:")
print(final_string[:300] + " ...")  # Short preview
