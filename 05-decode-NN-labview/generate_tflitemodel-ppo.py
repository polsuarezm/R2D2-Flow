import tensorflow as tf

# Define a PPO-style dummy model
model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,), name="observation"),
        tf.keras.layers.Dense(3, activation='relu'),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(7, activation='softmax', name="action_probs")
                    ])

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save to file
with open("ppo_policy_todecode.tflite", "wb") as f:
        f.write(tflite_model)

print("Dummy PPO TFLite model saved as 'ppo_policy_todecode.tflite'")

