#!/usr/bin/env python3
import socket
import time
import numpy as np
import json
import argparse

# ---------------------------------------
# Args
# ---------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--json_file", required=True)
args = parser.parse_args()

# ---------------------------------------
# Load parameters from config
# ---------------------------------------
with open(f"./conf/{args.json_file}", "r") as f:
    PARAMS = json.load(f)

# Networking (debug loopback recommended for local)
RECV_IP   = PARAMS["debug_ip"]          # where the sim RECEIVES weights (Python sends to udp_port_send)
RECV_PORT = PARAMS["udp_port_send"]
SEND_IP   = PARAMS["debug_ip"]          # where the sim SENDS trajectories (Python listens on udp_port_recv)
SEND_PORT = PARAMS["udp_port_recv"]

# Shapes / timing
SIZE_OBS_PER_UDP     = int(PARAMS["size_obs_array_per_UDP"])
TOTAL_DESCARTE_USED  = int(PARAMS["total_descarte_used"])
OBS_DIM              = SIZE_OBS_PER_UDP * (TOTAL_DESCARTE_USED + 1)

N_ACT       = int(PARAMS["size_actuator_array"])
EP_LEN      = int(PARAMS.get("episode_length", 50))
TOTAL_EPS   = int(PARAMS.get("total_episodes", 1000))
DELTA_T     = float(PARAMS.get("delta_t", 0.01))

# ---------------------------------------
# UDP sockets
# ---------------------------------------
# Receive weights from Python (it sends to crio_ip:udp_port_send)
sock_recv_w = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv_w.bind((RECV_IP, RECV_PORT))
sock_recv_w.settimeout(10.0)

# Send trajectories to Python (it listens on hp_ip:udp_port_recv)
sock_send_t = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"[SIM-OFFLOAD] Receiving weights on {RECV_IP}:{RECV_PORT}")
print(f"[SIM-OFFLOAD] Sending trajectories to {SEND_IP}:{SEND_PORT}")
print(f"[SIM-OFFLOAD] OBS_DIM={OBS_DIM}  N_ACT={N_ACT}  EP_LEN={EP_LEN}  TOTAL_EPS={TOTAL_EPS}")

rng = np.random.default_rng(1234)

def parse_weight_message(msg: str):
    """
    Expected single-datagram format from Python:
      '# arch; weights; identifier\\n'
      'OBS_H1_H2_..._ACTS;w1;w2;...;wN;IDENT'
    We just log the arch/identifier; we don't rebuild the net here.
    """
    try:
        lines = msg.splitlines()
        if not lines:
            return None, None
        if lines[0].startswith("#"):
            body = lines[1] if len(lines) > 1 else ""
        else:
            body = lines[0]
        parts = body.split(";")
        arch = parts[0] if parts else ""
        ident = parts[-1] if len(parts) >= 2 else ""
        return arch, ident
    except Exception:
        return None, None

def gen_state(prev_state, action, t):
    """
    Simple synthetic dynamics for debugging:
    - start from noise
    - add a small action-coupled bump to the 'most recent' chunk (last 4)
    NOTE: offloading mode expects a flat state vector of length OBS_DIM.
    """
    if prev_state is None:
        s = rng.normal(0, 0.1, size=OBS_DIM).astype(np.float32)
    else:
        s = prev_state.copy()
        s *= 0.95
        s += rng.normal(0, 0.02, size=OBS_DIM).astype(np.float32)

    # last chunk of 4 carries some action imprint
    last4 = s[-SIZE_OBS_PER_UDP:].copy()
    a0 = float(action[0]) if len(action) > 0 else 0.0
    x  = np.linspace(-1, 1, SIZE_OBS_PER_UDP, dtype=np.float32)
    bump = np.exp(-((x - np.clip(a0, -1, 1)*0.5)**2) / (2*0.25**2)) * 0.5
    last4 = 0.7*last4 + 0.3*(bump + rng.normal(0, 0.02, SIZE_OBS_PER_UDP))
    last4[-1] = np.clip(1.0 - abs(a0), -1.0, 1.0)  # reward proxy visible to you
    s[-SIZE_OBS_PER_UDP:] = last4.astype(np.float32)
    return s

def gen_action(t, arch_hint=None):
    """
    Dummy policy running on-device (CRIO). You can make this depend on the received weights.
    For now we use a smooth sinusoid per action dim.
    """
    a = np.zeros(N_ACT, dtype=np.float32)
    for i in range(N_ACT):
        a[i] = np.clip(np.sin(0.05 * t + i), -1.0, 1.0)
    return a

def gen_reward(state, action):
    # Simple correlation: prefer small |a0|
    base = 1.0 - abs(float(action[0])) if len(action) > 0 else 1.0
    noise = rng.normal(0, 0.02)
    return float(np.clip(base + noise, -1.0, 1.0))

def send_trajectory_one_episode(ep_idx, arch_hint):
    """
    Send EP_LEN packets:
      "s1,s2,...;a1,a2,...;r"
    then the terminator "<END>".
    """
    s_prev = None
    for t in range(EP_LEN):
        a = gen_action(t, arch_hint)
        s = gen_state(s_prev, a, t)
        r = gen_reward(s, a)

        s_prev = s

        s_str = ",".join(f"{v:.6f}" for v in s.tolist())
        a_str = ",".join(f"{v:.6f}" for v in a.tolist())
        r_str = f"{r:.6f}"
        payload = f"{s_str};{a_str};{r_str}"
        sock_send_t.sendto(payload.encode("utf-8"), (SEND_IP, SEND_PORT))

        # pacing
        time.sleep(DELTA_T)

    # terminator
    sock_send_t.sendto(b"<END>", (SEND_IP, SEND_PORT))

# ---------------------------------------
# Main loop
# ---------------------------------------
episode = 0

while episode < TOTAL_EPS:
    # 1) Wait for (updated) weights from Python each episode
    try:
        data, _ = sock_recv_w.recvfrom(65507)
        arch, ident = parse_weight_message(data.decode("utf-8", errors="ignore"))
        if arch:
            print(f"[SIM-OFFLOAD] Received weights: arch={arch} ident={ident}")
        else:
            print(f"[SIM-OFFLOAD] Received weights (unparsed length={len(data)})")
    except socket.timeout:
        print("[SIM-OFFLOAD] Waiting for initial weights timed out; continuing anyway...")

    # 2) Send one full trajectory
    episode += 1
    print(f"[SIM-OFFLOAD] Sending trajectory for episode {episode}/{TOTAL_EPS} ...")
    send_trajectory_one_episode(episode, arch_hint=arch if 'arch' in locals() else None)
    print(f"[SIM-OFFLOAD] Episode {episode} trajectory sent (then <END>).")

print("[SIM-OFFLOAD] Done.")

