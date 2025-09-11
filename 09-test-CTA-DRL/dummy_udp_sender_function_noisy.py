import socket
import time
import numpy as np
import json

# === Load parameters from config ===
with open("./conf/input_parameters_20250911.json", "r") as f:
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
obs = np.zeros(N_OBS_ARRAY, dtype=np.float32)
action = np.zeros(N_ACT_ARRAY, dtype=np.float32)
last_obs = 0.0

# === UDP Setup ===
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind((RECV_IP, RECV_PORT))
sock_recv.settimeout(DELTA_T*2)
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# === System Functions ===
def system_step_peak_spikes(
    action,
    peak_center=-0.5,
    peak_width=0.05,
    peak_amplitude=1.0,
    noise_level=0.05,
    peak_probability=0.5,
    sparsity=0.95  # <- higher means fewer non-zero values when no peak
):
    global t
    act = np.array(action[:N_OBS_ARRAY])

    # Default: zero signal
    signal = np.zeros(N_OBS_ARRAY, dtype=np.float32)

    if np.random.rand() < peak_probability:
        # Generate Gaussian peak
        x = act - peak_center
        peak = np.exp(-(x ** 2) / (2 * peak_width ** 2)) * peak_amplitude
        noise = np.random.normal(0, noise_level, size=N_OBS_ARRAY)
        signal = peak + noise
    else:
        # Generate sparse random noise burst
        sparse_noise = np.random.normal(0, noise_level, size=N_OBS_ARRAY)
        mask = np.random.rand(N_OBS_ARRAY) > sparsity  # e.g., 95% will be zero
        signal = sparse_noise * mask

    time.sleep(DELTA_T)
    return signal.tolist()


# === Fallbacks ===
def fallback_random_obs():
    return np.random.uniform(-1, 1, size=N_OBS_ARRAY).tolist()

def fallback_random_act():
    return np.random.uniform(-1, 1, size=N_ACT_ARRAY).tolist()

# === Main loop ===
print(f"Simulator listening on {RECV_IP}:{RECV_PORT}")
while True:
    counter += 1
    counter_miss = 0

    try:
        data, addr = sock_recv.recvfrom(1024)
        msg = data.decode().strip()
        parts = msg.split(";")
        timestamp = int(parts[0])

        action = [float(x) for x in parts[7:7 + N_ACT_ARRAY]]
        last_obs = system_step_peak_spikes(action,
                                           peak_center=5.0,
                                           peak_width=1.0,
                                           peak_amplitude=2.0,
                                           noise_level=1.0,
                                           peak_probability=0.05,
                                           sparsity=0.001
                                           )

        if counter == 1:
            obs[:] = last_obs[0]
        else:
            obs[0], obs[1], obs[2] = obs[1], obs[2], obs[3]
            obs[3] = last_obs[0]
            # Redefine first two values
            obs[0] = np.random.normal(loc=1.0, scale=0.01)     # tightly around 1
            obs[1] = np.random.uniform(low=-2.0, high=2.0)      # wide random

    except socket.timeout:
        timestamp = int(time.time() * 1000)
        print(f"missing - {counter}")

        try:
            action
        except NameError:
            action = fallback_random_act()

        last_obs = system_step_peak_spikes(action)

        if counter == 1:
            obs[:] = last_obs[0]
        else:
            obs[0], obs[1], obs[2] = obs[1], obs[2], obs[3]
            obs[3] = last_obs[0]

    except Exception as e:
        print(f"[ERROR] {e}")
        continue

    # === Format and Send ===
    if MESSAGE_TYPE == 1:
        response = f"{timestamp};" + ";".join(f"{v:.4f}" for v in obs) + ";" + ";".join(f"{v:.4f}" for v in action)
    else:
        response = f"{timestamp};" + "".join(str(int(v)) for v in obs)

    sock_send.sendto(response.encode(), (SEND_IP, SEND_PORT))

    if counter % 10 == 0:
        print(f"Recv action: {action} → Send obs: {response}", flush=True)
