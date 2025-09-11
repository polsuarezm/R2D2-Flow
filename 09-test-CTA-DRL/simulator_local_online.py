import socket
import time
import numpy as np
import json
from datetime import datetime

# === Load parameters from config ===
CONF_PATH = "./conf/input_parameters_20250911.json"  # <-- change to your JSON
with open(CONF_PATH, "r") as f:
    PARAMS = json.load(f)

# Networking (debug loopback)
RECV_IP   = PARAMS["debug_ip"]         # where the sim RECEIVES agent actions (agent sends to udp_port_send)
RECV_PORT = PARAMS["udp_port_send"]
SEND_IP   = PARAMS["debug_ip"]         # where the sim SENDS obs back (agent listens on udp_port_recv)
SEND_PORT = PARAMS["udp_port_recv"]

# Message / shapes
MESSAGE_TYPE         = int(PARAMS.get("message_type", 1))
SIZE_OBS_PER_UDP     = int(PARAMS["size_obs_array_per_UDP"])      # e.g., 4
TOTAL_DESCARTE       = int(PARAMS["total_descarte"])              # e.g., 1  -> packets per step = 2
TOTAL_DESCARTE_USED  = int(PARAMS["total_descarte_used"])         # env uses (used+1) * SIZE_OBS_PER_UDP
PACKETS_PER_STEP     = TOTAL_DESCARTE + 1
OBS_USED_TOTAL       = SIZE_OBS_PER_UDP * (TOTAL_DESCARTE_USED + 1)

N_ACT = int(PARAMS["size_actuator_array"])
DELTA_T = float(PARAMS.get("delta_t", 0.01))

# === UDP Setup ===
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind((RECV_IP, RECV_PORT))
sock_recv.settimeout(DELTA_T * 5)

sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"[SIM] Listening for actions on {RECV_IP}:{RECV_PORT}")
print(f"[SIM] Sending obs to        {SEND_IP}:{SEND_PORT}")
print(f"[SIM] PACKETS_PER_STEP={PACKETS_PER_STEP}  SIZE_OBS_PER_UDP={SIZE_OBS_PER_UDP}  OBS_USED_TOTAL={OBS_USED_TOTAL}")

# --- simple signal generator: spiky + noise ---
rng = np.random.default_rng(42)
t_global = 0

def gen_obs_block(action_vec, step_idx):
    """
    Generate one SIZE_OBS_PER_UDP-length block given the last action.
    Feel free to swap for whatever behavior you want to test.
    """
    # gaussian bump around action with noise
    x = np.linspace(-1, 1, SIZE_OBS_PER_UDP, dtype=np.float32)
    a = float(action_vec[0]) if len(action_vec) > 0 else 0.0
    peak_center = np.clip(a, -1, 1) * 0.5
    peak_width  = 0.25
    peak_amp    = 0.8
    noise_level = 0.05

    peak  = np.exp(-((x - peak_center) ** 2) / (2 * peak_width**2)) * peak_amp
    noise = rng.normal(0, noise_level, size=SIZE_OBS_PER_UDP).astype(np.float32)

    block = peak + noise
    # Let last element carry a "reward proxy" or any scalar you want visible in logs
    block[-1] = np.clip(1.0 - abs(a), -1.0, 1.0)
    return block.astype(np.float32)

def now_ms():
    return int(time.time() * 1000)

# --- main loop ---
step = 0
last_action = np.zeros(N_ACT, dtype=np.float32)

while True:
    try:
        # Receive one action packet from agent
        data, addr = sock_recv.recvfrom(4096)
        msg = data.decode().strip()
        parts = msg.split(";")
        # Expected from agent when MESSAGE_TYPE==1:
        # "<timestamp>;1;1;1;1;1;1;A0;A1;...;Ak"
        # Otherwise fallback to: "<timestamp>;A0A1...Ak"
        ts_in = int(parts[0])

        if MESSAGE_TYPE == 1:
            # 6 dummy flags, then actions
            action_fields = parts[7:7 + N_ACT]
            if len(action_fields) < N_ACT:
                # be lenient if fewer fields
                action_fields = action_fields + ["0.0"] * (N_ACT - len(action_fields))
            last_action = np.array([float(x) for x in action_fields], dtype=np.float32)
        else:
            # if you ever use compact form, parse accordingly
            # here we just keep previous action
            pass

        # Build the full observation used by env:
        # the env will read exactly (TOTAL_DESCARTE + 1) packets,
        # and inside each packet it only reads 4 floats (parts[1:5]).
        # We'll send PACKETS_PER_STEP packets with 4 floats each.
        ts_out = now_ms()
        for pkt in range(PACKETS_PER_STEP):
            block = gen_obs_block(last_action, step_idx=step*PACKETS_PER_STEP + pkt)

            # Ensure length is exactly SIZE_OBS_PER_UDP
            if block.shape[0] != SIZE_OBS_PER_UDP:
                block = np.resize(block, (SIZE_OBS_PER_UDP,)).astype(np.float32)

            payload = f"{ts_out};" + ";".join(f"{v:.6f}" for v in block[:SIZE_OBS_PER_UDP])
            sock_send.sendto(payload.encode(), (SEND_IP, SEND_PORT))

        step += 1

        if step % 10 == 0:
            print(f"[SIM] step={step:05d} recv_action={last_action.tolist()} -> sent {PACKETS_PER_STEP} packets (ts {ts_out})")

        # Simulate loop rate
        time.sleep(DELTA_T)

    except socket.timeout:
        # no action received in time; send some default obs anyway
        ts_out = now_ms()
        for pkt in range(PACKETS_PER_STEP):
            block = rng.normal(0, 0.1, size=SIZE_OBS_PER_UDP).astype(np.float32)
            payload = f"{ts_out};" + ";".join(f"{v:.6f}" for v in block)
            sock_send.sendto(payload.encode(), (SEND_IP, SEND_PORT))
        print("[SIM] timeout: sent default noise blocks")
        time.sleep(DELTA_T)

    except KeyboardInterrupt:
        print("\n[SIM] stopped by user.")
        break

    except Exception as e:
        print(f"[SIM][ERROR] {e}")
        time.sleep(0.1)
        continue

