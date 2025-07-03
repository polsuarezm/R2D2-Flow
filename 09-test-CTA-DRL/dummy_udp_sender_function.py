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
action = [0.0] * N_ACT_ARRAY
last_obs = 0.0

# === UDP Setup ===
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind((RECV_IP, RECV_PORT))
sock_recv.settimeout(0.5)
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# === System functions ===
def system_step_maxfun(action):
    global state, t
    act = np.array(action[:N_OBS_ARRAY])
    fun = (-act**4 + 3*act**3-act**2-2*act)/0.83
    noise = np.random.normal(0, 0.01, size=N_OBS_ARRAY)
    state = fun + noise
    time.sleep(DELTA_T)
    return state.tolist()

def fallback_random_obs():
    return np.random.uniform(-1, 1, size=N_OBS_ARRAY).tolist()

def fallback_random_act():
    return np.random.uniform(-1, 1, size=N_ACT_ARRAY).tolist()

# === Main loop ===
print(f"Simulator listening on {RECV_IP}:{RECV_PORT}")
while True:
    counter += 1

    # Shift obs history and update obs[3] later
    obs[0], obs[1], obs[2] = obs[1], obs[2], obs[3]

    try:
        data, addr = sock_recv.recvfrom(1024)
        msg = data.decode().strip()
        parts = msg.split(";")
        timestamp = int(parts[0])

        action = [float(x) for x in parts[7:7 + N_ACT_ARRAY]]
        last_obs = system_step_maxfun(action)
        obs[3] = last_obs[0]

    except socket.timeout:
        timestamp = int(time.time() * 1000)
        print(f"missing - {counter}")
        action = fallback_random_act()
        obs = fallback_random_obs()
        obs[3] = system_step_maxfun(action)[0]

    except Exception as e:
        print(f"[ERROR] {e}")
        continue

    # Format and send response
    if MESSAGE_TYPE == 1:
        response = f"{timestamp};" + ";".join(f"{v:.4f}" for v in obs) + ";" + ";".join(f"{v:.4f}" for v in action)
    else:
        response = f"{timestamp};" + "".join(str(int(v)) for v in obs)

    sock_send.sendto(response.encode(), (SEND_IP, SEND_PORT))

    if counter % 10 == 0:
        print(f"Recv action: {action} → Send obs: {response}", flush=True)
