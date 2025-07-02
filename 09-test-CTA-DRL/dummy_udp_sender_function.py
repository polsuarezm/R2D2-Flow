import socket
import time
import numpy as np
import json

# === Load parameters from config ===
with open("input_parameters_v1_debug.json", "r") as f:
    PARAMS = json.load(f)

RECV_IP = PARAMS["debug_ip"]
RECV_PORT = PARAMS["udp_port_send"]
SEND_IP = PARAMS["debug_ip"]
SEND_PORT = PARAMS["udp_port_recv"]

MESSAGE_TYPE = PARAMS["message_type"]
N_OBS_ARRAY = PARAMS["size_obs_array"]+1
N_ACT_ARRAY = PARAMS["size_actuator_array"]+1
counter = 0

# === UDP Setup ===
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind((RECV_IP, RECV_PORT))
sock_recv.settimeout(1.0)  # 1 second timeout
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# === Global state ===
t = 0.0
dt = 0.1
state = np.random.uniform(-1, 1, size=N_OBS_ARRAY)

def system_step(action):
    global state, t
    A = np.eye(N_OBS_ARRAY) * 0.9
    for i in range(N_OBS_ARRAY - 1):
        A[i, i + 1] = 0.1
    act = np.array(action[:N_OBS_ARRAY])
    nonlinear = 0.05 * np.sin(t + state)
    noise = np.random.normal(0, 0.01, size=N_OBS_ARRAY)
    state = A @ state + 0.1 * act + nonlinear + noise
    t += dt
    return state.tolist()

def fallback_random_obs():
    return np.random.uniform(-1, 1, size=N_OBS_ARRAY).tolist()

# === Main Loop ===
print(f"Simulator listening on {RECV_IP}:{RECV_PORT}")
while True:
    counter = counter + 1
    try:
        data, addr = sock_recv.recvfrom(2048)
        msg = data.decode().strip()
        parts = msg.split(";")
        timestamp = int(parts[0])
        action = [float(x) for x in parts[7:7 + N_ACT_ARRAY]]  # adapt if needed
        obs = system_step(action)
        print(f"Recv action: {action} → Send obs: {obs}")

    except socket.timeout:
        # No action received → fallback mode
        timestamp = int(time.time() * 1000)
        obs = fallback_random_obs()
        print(f"[timeout fallback] → Send random obs: {obs}")

    except Exception as e:
        print("Error:", e)
        time.sleep(0.1)
        continue

    # Send observation back
    if MESSAGE_TYPE == 1:
        response = f"{timestamp};" + ";".join(f"{v:.6f}" for v in obs) + f"{counter}"
    else:
        response = f"{timestamp};" + "".join(str(int(v)) for v in obs)

    sock_send.sendto(response.encode(), (SEND_IP, SEND_PORT))
    time.sleep(0.05)
