import socket
import time
import numpy as np
import tensorflow as tf
import struct


# --- case setup 
training = True
n_act    = 64

# === CONFIG ===
CRIO_IP = "172.22.10.2"
KV260_IP = "172.22.10.3"
UDP_PORT_SEND = 61557
UDP_PORT_RECV = 61555

# === Load TFLite Model ===
interpreter = tf.lite.Interpreter(model_path="ppo_policy_dummy.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

print("TFLite input shape:", input_shape)
print("TFLite output shape:", output_details[0]['shape'])

# === Setup UDP Sockets ===
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#sock_send.setblocking(1)

sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind((KV260_IP, UDP_PORT_RECV))
#sock_recv.setblocking(1)

print(f"Listening on {KV260_IP}:{UDP_PORT_RECV}...")

# === Loop === dummy variable just to try on deterministic episodes, random sequential calls
NUM_STEPS = 100000000000


for step in range(NUM_STEPS):
    #try:
        # === Receive observation from UDP ==
        #data_rcv = np.random.rand(1).astype(np.float32)
        start = time.perf_counter_ns()
        data_rcv, addr_rcv = sock_recv.recvfrom(32) ## waiting for 32 bits info from mass flux/ CTA
        #data_rcv_clean = data_rcv.decode().strip()
        start_conv = time.perf_counter_ns()
        data_rcv_clean = data_rcv.decode()
        print(data_rcv_clean)
        timestamp      = int(data_rcv_clean.split(";")[0])
        obs_str        = np.float32(data_rcv_clean.split(";")[1])
        end_conv = time.perf_counter_ns()
        #obs_str        = np.float32(obs_str)
        #print(f"{data_rcv_clean}")
        #obs_str = data_rcv_clean.decode().strip()

        #obs_str = data_rcv.decode()
        #obs_str = data_rcv

        #print(f"RECV>{obs_str}")

        #print(f"Received --> {data_rcv.decode()} // sent --> {obs_str} ")

        # === Parse observation ===
        #try:
        #    #obs_array = np.array([[float(float_str)]], dtype=np.float32)
        #    obs_array = np.fromstring(obs_str, sep=',', dtype=input_dtype).reshape(input_shape)
        #    #obs_array *= np.random.uniform(0.5,1.5)
        #except Exception as e:
        #    print(f"[Step {step}] Failed to parse observation: {obs_str}")
        #    continue

        # === Run Inference ===
        #t_start = time.time()

        # call to agent --> get action
        #interpreter.set_tensor(input_details[0]['index'], obs_array)
        start_tf = time.perf_counter_ns()
        interpreter.set_tensor(input_details[0]['index'], np.array([[obs_str]], dtype=np.float32))
        interpreter.invoke()

        action = interpreter.get_tensor(output_details[0]['index']).flatten()
        action_binary = (action>=0.5).astype(np.uint8) #now it has (64,)
        end_tf = time.perf_counter_ns()

        #t_end = time.time()

        #elapsed_ms = (t_end - t_start) * 1000000

        #print(f"[Step {step}] Received Obs: {obs_array.flatten()} -> Action: {action.flatten()} ({elapsed_ms:.2f} ms)")

        # === Send action back via UDP ===
        # need to convert to u64 output
        #bit_string = ''.join(str(b) for b in bit_array)
        #packed_int = int(bit_string, 2)
        #packed_bytes = struct.pack('>Q', packed_int)

        #print(f"send> {action_binary}")
        #start = time.perf_counter_ns()
        #sock_send.sendto(action_binary.tobytes(), (CRIO_IP, UDP_PORT_SEND))
        action_str = ''.join(str(b) for b in action_binary)
        message = f"{format(int(action_str,2), '016x')};{timestamp}"
        #message = f"{format(step*1000000, '016x')};{timestamp}"
        print(f"mess > {message}")
        #message = f"{format(step, '016x')};{timestamp}"
        sock_send.sendto(message.encode(), (CRIO_IP, UDP_PORT_SEND))
        #action_str = ",".join(map(str, action.flatten()))
        #sock_send.sendto(action_str.encode(), (CRIO_IP, UDP_PORT_SEND))
        end = time.perf_counter_ns()
        #time.sleep(0.05)  # Optional pacing

        print(f"total > {(end-start)*0.000001} ms -- tflite > {(end_tf-start_tf)*0.000001} ms")

    #except Exception as e:
     #   print(f"[Step {step}] Error: {e}")
      #  continue

# === Cleanup ===
sock_send.close()
sock_recv.close()

