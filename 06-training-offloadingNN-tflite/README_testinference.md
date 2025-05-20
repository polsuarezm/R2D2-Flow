# UDP Weight Sender

This script sends a text file containing neural network weights via UDP.

## 1. Requirements

- Python 3.x

activate venv for tensorflow

## 2. Configuration

Edit `udp_send_config.json`:

```json
{
  "weights_file": "weights_arch1.txt",
  "delay_seconds": 1,
  "repeat_count": 3
}
```

The destination IP and port are fixed in the script:

- IP: 172.22.10.2
- Port: 61557

## 3. Sending

Run the script:

```bash
cd inference_debug
python3 runUDP_inferencelabVIEW.py
```

## 4. Notes

- You can change `weights_file` to any of the generated weight files.
- Make sure the receiver is listening on the correct IP and port.

