import os
import numpy as np
import tensorflow as tf
import time 

MODEL_PATH = "ppo_policy_todecode.tflite"
OUTPUT_PATH = "model_dump.npz"

if not os.path.exists(MODEL_PATH):
    print(f"Model file not found: {MODEL_PATH}")
    exit(1)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, experimental_preserve_all_tensors=True)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("== Input details ==")
print("name:", input_details[0]['name'])
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])
print("\n== Output details ==")
print("name:", output_details[0]['name'])
print("shape:", output_details[0]['shape'])
print("type:", output_details[0]['dtype'])

# Try extracting weights and biases from the interpreter (works only if model includes weight tensors)
details = interpreter.get_tensor_details()
#print(f"{details}")

#time.sleep(10)
weights = {}
for d in details:
    name = d['name']
    print(f"{name}")
    time.sleep(1)
    tensor = interpreter.tensor(d['index'])()
    print(f"{tensor}")
    #weights[name] = tensor
    #weights[name] = interpreter.tensor(d['value'])
    print(f"Extracted tensor: {name}, shape: {tensor.shape}")

expected_keys = ['dense_0/kernel', 'dense_0/bias',
        'dense_1/kernel', 'dense_1/bias',
        'dense_2/kernel', 'dense_2/bias'
        ]

print(f"{expected_keys}")

#if all(k in weights for k in expected_keys):
np.savez(OUTPUT_PATH, **weights)
print(f"Model weights dumped to {OUTPUT_PATH}")
print(f"")
# === Deterministic math: Manual forward pass ===
def relu(x):
    return np.maximum(0, x)
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

# Input observation
sample_obs = np.array([[0.5]], dtype=np.float32)

# Layer weights
W1 = weights['serving_default_observation:0/kernel']
b1 = weights['serving_default_observation:0/bias']
W2 = weights['dense_1/kernel']
b2 = weights['dense_1/bias']
W3 = weights['dense_2/kernel']
b3 = weights['dense_2/bias']

# Forward pass
z1 = relu(np.dot(sample_obs, W1) + b1)
z2 = relu(np.dot(z1, W2) + b2)
output = softmax(np.dot(z2, W3) + b3)

#TODO: SKIP NONLIN FUNCS (RELU OR SIGM)

print("\nSample input:", sample_obs.flatten())
print("Deterministic output (manual forward pass):", output.flatten())
