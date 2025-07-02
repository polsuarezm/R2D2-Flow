import socket
import time
import numpy as np
import json

# === Load parameters from the same config to stay consistent ===
with open("input_parameters_v1.json", "r") as f:
    PARAMS = json.load(f)

UDP_IP = PARAMS["debug_ip"]
UDP_PORT = PARAMS["udp_port_recv"]
MESSAGE_TYPE = PARAMS["message_type"]
N_OBS_ARRAY = PARAMS["size_obs_array"]

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def generate_random_observation():
    # Define ranges per observation index, for example:
    ranges = [
        (0, 1),     # obs[0]
        (-5, 5),    # obs[1]
        (0, 100),   # obs[2]
        (10, 20),   # obs[3]
        (0, 1)      # obs[4] (action feedback)
    ]
    return [np.random.uniform(low, high) for (low, high) in ranges[:N_OBS_ARRAY+1]]

while True:
    timestamp = int(time.time() * 1000)  # millisecond timestamp
    obs_values = generate_random_observation()

    if MESSAGE_TYPE == 1:
        message = f"{timestamp};" + ";".join(f"{v:.6f}" for v in obs_values)
    else:
        message = f"{timestamp};{''.join(str(int(v)) for v in obs_values)}"

    sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
    print("Sent:", message)
    time.sleep(0.10)  # send every 50 ms
