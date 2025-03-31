import tensorflow as tf

# Define a PPO-style dummy model
model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,), name="observation"),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(64, activation='softmax', name="action_probs")
                    ])

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save to file
with open("ppo_policy_dummy.tflite", "wb") as f:
        f.write(tflite_model)

print("Dummy PPO TFLite model saved as 'ppo_policy_dummy.tflite'")

