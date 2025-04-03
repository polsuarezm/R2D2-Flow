#!/bin/bash

# === CPU Monitoring and Execution Helper for KV260 DRL Agent ===
# Use this script to pin the agent to a specific core, limit TensorFlow threads,
# and monitor CPU usage to prevent overload and latency issues with CRIO.

# Path to your venv activation script
source /home/polete/Documents/DRL-tflite-KV260-env/bin/activate

# Set environment variables for TensorFlow threading (limit to 1 thread)
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

# Launch your agent script pinned to CPU core 1
# Replace "run_agent.py" with your actual script filename
echo "Launching agent on CPU core 1..."
#taskset -c 1 python3 run_training_UDP_tflite.py
taskset -c 1 python3 run_training_offloadUDP_tf_OPTIMIZED.py
#AGENT_PID=$!

# Monitor the CPU usage of the agent every 0.5s using top
# You can press 'q' to exit the top view
#echo "Monitoring CPU usage of PID $AGENT_PID..."
#top -d 0.5 -p $AGENT_PID

