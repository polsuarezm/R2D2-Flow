# RT-RPO / PPO with LabVIEW / cRIO Integration

This repository contains a PyTorch RPO/PPO implementation that:

- Communicates with a LabVIEW / cRIO real-time target via TCP  
- Receives batched trajectories (`BATCH ... END`)  
- Optionally deploys neural-network actor weights back to the cRIO  
- Trains a continuous-action policy (actor–critic)
- Logs training to TensorBoard and a CSV file

The main script:
- `SHARP_rpo_continuous_action_v3.py`

Optional live plotting script:
- `plot_step_log_live.py`



---

# 1. Prerequisites

- **Python 3.10**
- **Visual Studio Code**
- Python extension for VS Code (Microsoft)
- PyTorch installed (CPU or GPU)
- (Optional) Matplotlib for live plotting


---

# 2. Create & Activate a Virtual Environment

Open a terminal in your project folder.

### Create the venv
```bash
python -m venv .venv
```

### Activate on Windows (cmd)
```bash
.venv\Scripts\activate
```

### Activate on Windows PowerShell
```powershell
.\.venv\Scripts\Activate.ps1
```

### Activate on Linux/macOS
```bash
source .venv/bin/activate
```

Your terminal will show `(.venv)` when active.



---

# 3. Install Dependencies

With the virtual environment active:

```bash
pip install --upgrade pip
pip install numpy torch tyro tensorboard
```

For live plotting:

```bash
pip install matplotlib
```



---

# 4. Open in VS Code & Select Interpreter

### Launch VS Code:
```bash
code .
```

Then in VS Code:

- Press `Ctrl+Shift+P`
- Select **Python: Select Interpreter**
- Choose the interpreter inside `.venv` (usually `.venv\Scripts\python.exe`)



---

# 5. Run or Debug the Script

### Run normally:
```bash
python SHARP_rpo_continuous_action_v3.py
```

### Debug configuration

Create `.vscode/launch.json` (if not present):

```jsonc
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug RT-RPO (default)",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/SHARP_rpo_continuous_action_v3.py",
      "console": "integratedTerminal",
      "justMyCode": true,
      "args": []
    }
  ]
}
```

Then press **F5** to debug with breakpoints.



---

# 6. How the System Works (High Overview)

1. Script starts -> initializes agent, optimizer, logging, rollout buffers.  
2. Connects to LabVIEW/cRIO via TCP:
   - `"RT waiting for deployment"` → send NN actor parameters  
   - `"RT testing NN"` → wait  
   - `"BATCH ... END"` → receive trajectories
3. Parses the batch into tensors:
   - Observations  
   - Actions (clean)  
   - Actions (noisy, executed)  
   - Logprobs (from RT)  
   - Reward (delayed by one step)
4. Builds a rollout buffer and computes GAE.
5. Performs PPO/RPO optimization.
6. Logs metrics.
7. Saves checkpoints periodically.
8. Loops again.



---

# 7. Important Command-Line Arguments

All parameters live in the `Args` dataclass and can be overridden via CLI using **tyro**.

### Training configuration

| Argument | Meaning | Default |
|---------|---------|---------|
| `--total_timesteps` | Total timesteps to compute number of updates | 8M |
| `--learning_rate` | Adam LR | 0.003 |
| `--num_steps` | Batch size coming from RT | 50 |
| `--gamma` | Discount factor | 0.7 |
| `--gae_lambda` | Advantage estimation lambda | 0.95 |
| `--num_minibatches` | Minibatches per PPO update | 1 |
| `--update_epochs` | PPO update epochs | 5 |
| `--clip_coef` | PPO clipping | 0.2 |
| `--ent_coef` | Entropy loss coefficient | 0.0 |
| `--vf_coef` | Value loss coefficient | 0.5 |
| `--rpo_alpha` | RPO noise to mean action (0 = PPO) | 0.0 |

### Checkpoints & loading

| Argument | Meaning |
|----------|----------|
| `--checkpoint_dir` | Base directory for saving `.pt` files |
| `--save_interval` | Save model every N updates |
| `--load_model_path` | Load a checkpoint at start |

**Example:**

```bash
python SHARP_rpo_continuous_action_v3.py --load_model_path checkpoints/myrun/model_update_000200.pt
```



---

# 8. Logging Outputs

### TensorBoard logs:
```
runs/<run_name>/
```

Start TensorBoard:
```bash
tensorboard --logdir runs
```

### CSV step logs:
```
runs/<run_name>/step_log.csv
```

Contains per-step:
- global step
- update number
- step within update
- reward
- noisy action
- mean action
- value estimate



---

# 9. Live Plotting (Optional)

Use:

```bash
python plot_step_log_live.py
```

Features:

- Refreshes automatically every 10 seconds
- Plots:
  1. Reward vs. step  
  2. Reward vs. noisy action  
  3. Value vs. step  
  4. Action (mean & noisy) vs. step  
- No pandas required  
- Non-blocking single window



---

# 10. Notes on TCP Communication Protocol

The RT/cRIO side must send messages like:

- `RT waiting for deployment`
- `RT testing NN`
- `BATCH ... END` with trajectory data

The Python side will:

- Send neural network parameters when deployment requested  
- Acknowledge batches using `DATA RECEIVED`  
- Reconstruct split batches automatically  
- Validate data before training  



---

# 11. Example Full Command

```bash
python SHARP_rpo_continuous_action_v3.py \
    --total_timesteps 400000 \
    --gamma 0.7 \
    --save_interval 50
```



---

If you need a **deploy-only script** (no training, just load model + send it to cRIO), tell me and I can generate it.  

