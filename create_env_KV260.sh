#!/bin/bash

# Crear y activar el entorno virtual
python3 -m venv kv260_env
source kv260_env/bin/activate

# Actualizar pip y setuptools
pip install --upgrade pip setuptools

# Instalar TensorFlow Lite
pip install tflite-runtime

# Instalar TensorForce
pip install tensorforce

# Instalar Stable-Baselines3 y dependencias
pip install stable-baselines3[extra] gym numpy

# Verificar instalaciones
python3 -c "import tflite_runtime.interpreter; print('TensorFlow Lite instalado')"
python3 -c "import tensorforce; print('TensorForce instalado')"
python3 -c "import stable_baselines3; print('Stable-Baselines3 instalado')"

echo "Configuración completada. Usa 'source kv260_env/bin/activate' para activar el entorno."