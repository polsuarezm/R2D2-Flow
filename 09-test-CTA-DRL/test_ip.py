import socket

# === Configuration ===
PC_IP = "172.22.11.1"           # Local IP
UDP_PORT_RECV = 61555           # Listening port
CRIO_IP = "172.22.11.2"         # Target (cRIO) IP
UDP_PORT_SEND = 61557           # Port to send back to cRIO

# === Setup sockets ===
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind((PC_IP, UDP_PORT_RECV))

sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"📡 Listening on {PC_IP}:{UDP_PORT_RECV}... expecting 5 values, echoing back timestamp")

try:
    while True:
        data, addr = sock_recv.recvfrom(2048)
        msg = data.decode().strip()
        fields = msg.split(";")
        print(f"\n📥 Received from {addr}: {msg}")

        if len(fields) == 5:
            timestamp = fields[0]
            print(f"✅ Parsed timestamp: {timestamp}")

            # Echo back the timestamp
            response = f"{timestamp}"
            sock_send.sendto(response.encode(), (CRIO_IP, UDP_PORT_SEND))
            print(f"📤 Sent back to {CRIO_IP}:{UDP_PORT_SEND}: {response}")
        else:
            print(f"⚠️ Unexpected format: {len(fields)} fields (expected 5)")

except KeyboardInterrupt:
    print("\n🛑 Stopped.")
finally:
    sock_recv.close()
    sock_send.close()

