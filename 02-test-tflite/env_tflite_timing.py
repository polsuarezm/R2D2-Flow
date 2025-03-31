import time
import numpy as np
import tensorflow as tf

# Load the TFLite model (replace 'model.tflite' with your real model path)
interpreter = tf.lite.Interpreter(model_path="ppo_policy_dummy.tflite")
interpreter.allocate_tensors()

# Get input/output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print input/output details (optional for debug)
print("Input shape:", input_details[0]['shape'])
print("Output shape:", output_details[0]['shape'])

# Simulate reading a single observation (e.g., from sensor)
def read_observation():
    # Dummy observation: replace this with real SPI data
    return np.random.rand(*input_details[0]['shape']).astype(np.float32)

# Run the loop
NUM_EPISODES = 100

for step in range(NUM_EPISODES):
    # Start timin
    t_start = time.time()
    # Simulate observation
    obs = read_observation()
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], obs)

    # Run inference
    interpreter.invoke()
 
    # Get output tensor (the action)
    action = interpreter.get_tensor(output_details[0]['index'])
 
    # End timing
    t_end = time.time()
    elapsed_ms = (t_end - t_start) * 1000

    print(f"[Step {step}] Action: {action}, Inference Time: {elapsed_ms:.3f} ms")

