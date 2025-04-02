import os
import numpy as np
import tensorflow as tf

MODEL_PATH = "ppo_policy_todecode.tflite"
#MODEL_PATH = "/scratch/polsm/011-DRL-experimental/AFC-DRL-experiment/04-training-UDP-tflite/ppo_policy_todecode.tflite"
OUTPUT_PATH = "model_dump.npz"

if not os.path.exists(MODEL_PATH):
    print(f"Model file not found: {MODEL_PATH}")
    exit(1)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Try extracting weights and biases from the interpreter (works only if model includes weight tensors)
try:
    details = interpreter.get_tensor_details()
    weights = {}
    for d in details:
        name = d['name']
        if "dense" in name and ("kernel" in name or "bias" in name):
            tensor = interpreter.tensor(d['index'])()
            weights[name] = tensor
            print(f"Extracted tensor: {name}, shape: {tensor.shape}")

    expected_keys = [
        'dense/kernel', 'dense/bias',
        'dense_1/kernel', 'dense_1/bias',
        'dense_2/kernel', 'dense_2/bias'
    ]

    if all(k in weights for k in expected_keys):
        # Save all weights by name for easier use in external environments
        np.savez(OUTPUT_PATH, **weights)
        print(f"Model weights dumped to {OUTPUT_PATH}")

        # === Do forward pass manually from weights ===
        def relu(x):
            return np.maximum(0, x)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Sample input to test deterministic math
        sample_obs = np.array([[0.5]], dtype=np.float32)

        # Extract named layers
        W1 = weights['dense/kernel']
        b1 = weights['dense/bias']
        W2 = weights['dense_1/kernel']
        b2 = weights['dense_1/bias']
        W3 = weights['dense_2/kernel']
        b3 = weights['dense_2/bias']


        # Manual forward pass: obs -> Dense -> Dense -> Dense -> sigmoid
        layer1 = relu(np.dot(sample_obs, W1) + b1)
        layer2 = relu(np.dot(layer1, W2) + b2)
        output = sigmoid(np.dot(layer2, W3) + b3)

        print("\nSample input:", sample_obs.flatten())
        print("Output (manual forward pass):")
        print(output.flatten())

    else:
        print("Warning: Did not find all expected tensors for the 3 dense layers")

except Exception as e:
    print("Error while extracting weights:", str(e))
