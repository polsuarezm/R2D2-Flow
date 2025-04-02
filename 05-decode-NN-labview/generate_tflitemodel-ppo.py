import tensorflow as tf

# Define a PPO-style dummy model
model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(5,), name="observation"),
        tf.keras.layers.Dense(7, activation='relu'),
        tf.keras.layers.Dense(7, activation='relu'),
        tf.keras.layers.Dense(11, activation='softmax', name="action_probs")
                    ])

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save to file
with open("ppo_policy_todecode.tflite", "wb") as f:
        f.write(tflite_model)

print("Dummy PPO TFLite model saved as 'ppo_policy_todecode.tflite'")

