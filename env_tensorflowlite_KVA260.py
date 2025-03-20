import tflite_runtime.interpreter as tflite
import numpy as np

# Cargar el modelo TFLite
model_path = "modelo.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Obtener información sobre los tensores de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Crear una entrada de prueba (suponiendo que el modelo espera un tensor de 1x224x224x3)
input_shape = input_details[0]['shape']
input_data = np.random.rand(*input_shape).astype(np.float32)

# Cargar la entrada en el modelo
interpreter.set_tensor(input_details[0]['index'], input_data)

# Ejecutar la inferencia
interpreter.invoke()

# Obtener los resultados de la salida
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Resultados de inferencia:", output_data)