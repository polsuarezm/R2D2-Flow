Ejecución de Stable-Baselines3 en KV260

Este documento describe los pasos para configurar y ejecutar Stable-Baselines3 en la AMD Xilinx Kria KV260.

1. Preparar la KV260

1.1. Instalar dependencias

Conéctate a la KV260 mediante SSH o abre una terminal directamente en la placa. Luego, instala las dependencias necesarias:

Crea y activa un entorno virtual:

1.2. Instalar Stable-Baselines3 y dependencias

Instala Stable-Baselines3 y las librerías necesarias:

Verifica la instalación:

Si hay problemas, revisa la compatibilidad con la arquitectura ARM.

2. Configurar el entorno de entrenamiento

Asegúrate de tener los permisos correctos para acceder al hardware:

Reinicia la KV260 para aplicar los cambios:

3. Ejecutar el script de entrenamiento en la KV260

Ubícate en la carpeta donde guardaste el script y ejecútalo:

Si todo está bien, deberías ver los mensajes de entrenamiento y evaluación del modelo.

4. Optimización para la KV260

Si necesitas mayor rendimiento, puedes considerar:

Usar versiones optimizadas de Gym: Algunas versiones incluyen soporte para aceleración por hardware.

Reducir la carga computacional: Entrenar modelos más pequeños o usar menos pasos por iteración.

Delegados de hardware: Si la KV260 soporta aceleración para cálculos de IA, puedes configurarlo en el entorno.

5. Conectar el agente a un bus de datos (Opcional)

Si necesitas conectar el script al bus de datos de 1 GHz, asegúrate de que tienes un driver o acceso a la API adecuada. Un ejemplo de lectura de datos en Python:

⚠️ IMPORTANTE: Dependiendo de la configuración del bus, es posible que necesites permisos especiales o acceso root para leer los datos.