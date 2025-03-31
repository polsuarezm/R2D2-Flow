import socket 
import time 

#UDP_IP   = "0.0.0.0"
UDP_PORT = 61555
UDP_IP   = "172.22.10.3" #IP del CRIO

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP,UDP_PORT))

print(f"Listening on {UDP_PORT}...")

while True:
    data, addr = sock.recvfrom(32)
    receive_time = time.perf_counter()

    print(f"Received from {addr} --> {data.decode()} ")


    #sock.sendto(data, addr)
    
