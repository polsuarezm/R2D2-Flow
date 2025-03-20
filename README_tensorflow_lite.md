Ejecución de TensorFlow Lite en KV260

Este documento describe los pasos para configurar y ejecutar TensorFlow Lite en la AMD Xilinx Kria KV260.

1. Preparar la KV260

1.1. Instalar dependencias

Conéctate a la KV260 mediante SSH o abre una terminal directamente en la placa. Luego, instala las dependencias necesarias:

Instala las herramientas de Xilinx:

1.2. Instalar TensorFlow Lite

TensorFlow Lite en la KV260 debe ser compatible con la arquitectura ARM. Se recomienda instalar TFLite Runtime en lugar de la versión completa de TensorFlow:

Verifica la instalación:

Si hay problemas, revisa si necesitas una versión específica optimizada para ARM.

2. Configurar el entorno de TensorFlow Lite

Asegúrate de tener los permisos correctos para acceder al hardware:

Reinicia la KV260 para aplicar los cambios:

3. Ejecutar el script en la KV260

Ubícate en la carpeta donde guardaste el script y ejecútalo:

Si todo está bien, deberías ver los mensajes de ejecución de TensorFlow Lite.

4. Optimización para la KV260

Si necesitas mayor rendimiento, puedes usar aceleración por hardware:

Compilar modelos para EdgeTPU o FPGA:
Puedes optimizar modelos utilizando herramientas de Vitis AI o TensorFlow Model Optimization Toolkit.

Delegados para aceleración por hardware:
Si la KV260 tiene soporte para delegados específicos (como OpenCL o NPU), puedes usarlos en TensorFlow Lite.

5. Conectar el script al bus de datos (Opcional)

Si necesitas conectar el script al bus de datos de 1 GHz, asegúrate de que tienes un driver o acceso a la API adecuada. Un ejemplo de lectura de datos en Python:

IMPORTANTE: Dependiendo de la configuración del bus, es posible que necesites permisos especiales o acceso root para leer los datos.