Ejecución de TensorFlow en KV260

Este documento describe los pasos para configurar y ejecutar TensorFlow en la AMD Xilinx Kria KV260.

1. Preparar la KV260

1.1. Instalar dependencias

Conéctate a la KV260 mediante SSH o abre una terminal directamente en la placa. Luego, instala las dependencias necesarias:

Instala las herramientas de Xilinx:

1.2. Instalar TensorFlow

TensorFlow en la KV260 debe ser compatible con la arquitectura ARM. Se recomienda instalar una versión optimizada:

Verifica la instalación:

Si hay problemas, revisa si necesitas una versión específica optimizada para ARM.

2. Configurar el entorno de TensorFlow

Si estás utilizando TensorForce, instálalo también:

Asegúrate de tener los permisos correctos para acceder al hardware:

Reinicia la KV260 para aplicar los cambios:

3. Ejecutar el script en la KV260

Ubícate en la carpeta donde guardaste el script y ejecúta:

Si todo está bien, deberías ver la versión de TensorFlow y los mensajes de ejecución.

4. Optimización para la KV260 (Opcional)

Si necesitas mayor rendimiento, puedes usar:

TensorFlow Lite:

Luego, usa modelos convertidos a TFLite.

Offloading a FPGA:
La KV260 tiene una FPGA programable. Para usar aceleración por hardware, puedes compilar modelos con Vitis AI.

5. Conectar el script al bus de datos (Opcional)

Si necesitas conectar el script al bus de datos de 1 GHz, asegúrate de que tienes un driver o acceso a la API adecuada. Un ejemplo de lectura de datos en Python:

IIMPORTANTE: Dependiendo de la configuración del bus, es posible que necesites permisos especiales o acceso root para leer los datos.