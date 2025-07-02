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
N_OBS_ARRAY = PARAMS["size_obs_array"]
N_ACT_ARRAY = PARAMS["size_actuator_array"]
DELTA_T = PARAMS["delta_t"]

# === State initialization ===
counter = 0
t = 0.0
dt = 0.1
state = np.random.uniform(-1, 1, size=N_OBS_ARRAY)
obs = np.zeros(N_OBS_ARRAY, dtype=np.float32)
action = [0.0] * N_ACT_ARRAY  # ensure it's always a list

# === UDP Setup ===
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind((RECV_IP, RECV_PORT))
sock_recv.settimeout(0.15)
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# === System functions ===
def system_step_maxfun(action):
    global state, t
    act = np.array(action[:N_OBS_ARRAY])
    fun = (-act**4 + act**3) * 10
    noise = np.random.normal(0, 0.01, size=N_OBS_ARRAY)
    state = fun + noise
    time.sleep(DELTA_T)
    return state.tolist()

def fallback_random_obs():
    return np.random.uniform(-1, 1, size=N_OBS_ARRAY).tolist()

# === Main loop ===
print(f"Simulator listening on {RECV_IP}:{RECV_PORT}")
while True:
    counter += 1
    try:
        data, addr = sock_recv.recvfrom(1024)
        msg = data.decode().strip()
        parts = msg.split(";")
        timestamp = int(parts[0])

        action = [float(x) for x in parts[7:7 + N_ACT_ARRAY]]

        last_obs = system_step_maxfun(action)

        # Shift obs history and update obs[3]
        obs[0], obs[1], obs[2] = obs[1], obs[2], obs[3]
        obs[3] = last_obs[0]

    except socket.timeout:
        timestamp = int(time.time() * 1000)
        print(f"missing - {counter}")
        obs = fallback_random_obs()
        action = [0.0] * N_ACT_ARRAY  # dummy action
    except Exception as e:
        print(f"[ERROR] {e}")
        continue

    # Format and send response
    if MESSAGE_TYPE == 1:
        response = f"{timestamp};" + ";".join(map(str, obs)) + ";" + ";".join(map(str, action))
    else:
        response = f"{timestamp};" + "".join(str(int(v)) for v in obs)

    sock_send.sendto(response.encode(), (SEND_IP, SEND_PORT))
    print(f"Recv action: {action} → Send obs: {response}")
