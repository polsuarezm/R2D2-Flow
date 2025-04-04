# Tutorial: Running the DRL + Experimental Environment on Ubuntu

This document provides step-by-step instructions for setting up and executing the AFC-DRL experimental environment on an Ubuntu-based laptop.

---

## 1. Accessing the Main Project Directory

If the repository has not yet been downloaded, clone it using:

```bash
git clone https://github.com/polsuarezm/AFC-DRL-experiment.git
cd AFC-DRL-experiment
```

---

## 2. Setting Up the Python Virtual Environment

If the virtual environment does not exist, it must be created and configured. A shell script is provided to automate this process:

```bash
. create_py-venv_tf_ubuntu.sh
```

This script installs the required Python environment along with TensorFlow and other dependencies.

Once inside the project folder, the Python environment should activate automatically via `.bashrc`. If it does not, it can be activated manually:

```bash
source pyenv-DRL-tf/bin/activate
```

## 3. Synchronising github repo 

To update the folder from the repo to the local:
```bash
. pull_git_guardiola.sh
```

To push the local files modifications to the repo:
```bash
. push_git_guardiola.sh
```
---

## 3. Running Tests

### 3.1 Test via UDP (Control and Learning Loop in Python)

This test executes both the control loop and the learning loop in a unified Python script, which communicates with the CRIO via UDP.

```bash
cd 04-training-UDP-tflite
python3 run_training_UDP_tflite_OPTIMIZED.py
```

**Note:**  
- Ensure that the JSON configuration file is updated with the correct IP address, port number, and other communication parameters.
- This script automatically clears any previous logs.

---

### 3.2 Test with Offloaded Neural Network (Inference on CRIO)

This test splits execution: the neural network inference runs on the CRIO, while the training loop (using PPO) executes on the host PC. Both processes communicate via UDP.

```bash
cd 06-training-offloadingNN-tflite
python3 run_training_offloadUDP_tf_OPTIMIZED.py
```

**Note:**  
- Confirm that the JSON file contains correct communication parameters (IP, port, etc.).
- The UDP connection transfers neural network weights, biases, and learning buffers (e.g., `<state, action, reward>`).

---








