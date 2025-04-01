# DRL UDP Agent on KV260 with CRIO Interface

This project runs a Deep Reinforcement Learning (DRL) inference + training loop on the KV260 board using a TFLite model. It communicates with a National Instruments CRIO over UDP, sending binary actions and receiving observations.

## 🚀 Features
- TFLite inference loop
- Experience collection and episode-based training
- Customizable model architecture
- UDP I/O with LabVIEW/CRIO
- Debugging + timing logs
- Weight and loss tracking
- TensorBoard and loss plots
- Configurable via `config.json`

---

## 📄 File: `config.json`
```json
{
  "training": true,
  "n_actions": 64,
  "episode_length": 1000,
  "total_episodes": 10,
  "hidden_units": 64,
  "model_path": "ppo_policy_dummy.tflite",
  "log_dir_template": "logs/ppo_run_{}",
  "crio_ip": "172.22.10.2",
  "kv260_ip": "172.22.10.3",
  "udp_port_send": 61557,
  "udp_port_recv": 61555,
  "debug": true
}
```

---

## 🛠️ Performance Tuning / Debug Notes

### 1. CPU Overload on KV260?
If you're seeing latency between KV260 and CRIO, it's likely due to CPU overload.

### ✅ Solutions:
- **Add delay between loop steps**:
  ```python
  time.sleep(0.001)  # sleep 1 ms
  ```
- **Limit TensorFlow threads**:
  ```python
  import os
  os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
  os.environ["TF_NUM_INTEROP_THREADS"] = "1"
  ```
  Put this **before** importing `tensorflow`.

- **Pin script to CPU core**:
  ```bash
  taskset -c 1 python3 run_agent.py
  ```

- **Avoid file logging unless debugging**:
  Set `"debug": false` in `config.json` to disable logging.

- **Monitor with top**:
  ```bash
  top -d 0.5 -p $(pgrep -f run_agent.py)
  ```

### 2. Timing Logs
When `debug = true`, a `debug_log.txt` file is created with timing info:
```
[TIMING] Step 0: total=1.15 ms, inference=0.36 ms
[TIMING] Model update duration: 248.72 ms
```

---

## Output Files
- `debug_log.txt`: step-by-step latency logs
- `loss_curve.png`: visual loss curve
- `weights_stepX_layerY.npy`: model weights for each layer per episode

---

## Requirements
- Python 3.8+
- TensorFlow Lite runtime
- NumPy, Matplotlib
- KV260 running Linux with UDP access

---

## Training Model
Training uses episode memory and performs batch updates at the end of each episode using:
```python
loss = model.fit(obs_batch, act_batch, epochs=5)
```
Weights are converted to `.tflite` after each episode and saved.

---

## Monitoring
Use TensorBoard:
```bash
tensorboard --logdir logs/
```

---

## UDP Format
- **Received from CRIO**: `timestamp;float_value`
- **Sent to CRIO**: `hex_action;timestamp` (64-bit binary action packed as hex)

---

## Support
Having trouble? Want to optimize CPU scheduling or async I/O? Message or open a new issue!

