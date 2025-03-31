import socket 
import time 
import csv
import struct

CRIO_IP = "172.22.10.2"
kv260_IP = "172.22.10.3"

UDP_PORT_send = 61557
UDP_PORT_recv = 61555

NUM_PINGS = 100000000 
csv_log = "latency_log.csv"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#sock.settimeout(1.0)
sock.setblocking(1)

sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind((kv260_IP,UDP_PORT_recv))
#sock_recv.settimeout(1.0)
sock_recv.setblocking(1)


for i in range(NUM_PINGS):
    send_time = time.perf_counter_ns()
    #message = f"{send_time} ms"
    message = int("f", 32)
    message = struct.pack('>Q', message)
    #message = f"{i};{send_time:.10f}" #Format: ping_number; sendtimestamp

    start = time.perf_counter_ns()
    #sock.sendto(message.encode(), (CRIO_IP, UDP_PORT_send))
    sock.sendto(message, (CRIO_IP, UDP_PORT_send))

    print(f"Listening on {UDP_PORT_send}...")

    data_rcv, addr_rcv = sock_recv.recvfrom(40)
    receive_time = time.perf_counter_ns()
    print(f"step>{i} sent>{message} Recv>{data_rcv.decode()}")


    end = time.perf_counter_ns()
    rtt = (end-start)*0.000001

    #time.sleep(1.0)

    print(f"PING {i} - RTT = {rtt:.6f} ms")

sock.close()
