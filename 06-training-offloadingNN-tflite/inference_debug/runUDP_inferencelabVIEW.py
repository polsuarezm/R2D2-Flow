import socket
import json
import time

# === Static IP and Port from known config ===
TARGET_IP = "172.22.10.2"       # crio_ip
TARGET_PORT = 61557             # udp_port_send

# === Load JSON config ===
with open("udp_send_config.json", "r") as f:
    config = json.load(f)

weights_file = config["weights_file"]
delay = config.get("delay_seconds", 1)
repeat = config.get("repeat_count", 3)

# === Read payload ===
with open(weights_file, "r") as f:
    payload = f.read().strip()

# === UDP Send Loop ===
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(f"Sending '{weights_file}' to {TARGET_IP}:{TARGET_PORT} ...")

for i in range(repeat):
    sock.sendto(payload.encode(), (TARGET_IP, TARGET_PORT))
    print(f"[{i+1}/{repeat}] Sent {len(payload)} bytes")
    time.sleep(delay)

sock.close()
print("✅ Done.")

