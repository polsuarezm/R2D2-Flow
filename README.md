# R2D2-Flow: Real-time Reinforcement learning for Drag reduction and Dynamics control in Flow control experiments. 

UDP-based reinforcement learning framework for PPO with support for:

1. **Online Training** — PPO runs in Python, sends actions via UDP, receives observations, and learns.
2. **Online Inference** — PPO runs in Python, sends actions via UDP, receives observations, no learning.
3. **Offloading Training** — CRIO/KV260 executes the policy; Python receives full trajectories, trains the actor, and sends updated weights back over UDP.
4. **Offload Inference** — CRIO/KV260 executes the policy; Python only sends initial weights and logs trajectories (no update).

---

## 1. Environment Setup

Clone the repo and create a virtual environment:

```bash
git clone <your_repo_url>
cd <your_repo>

make venv          # create .venv (Python 3.9+ recommended)
source .venv/bin/activate
make install       # install dependencies
````

Dependencies installed:

* [stable-baselines3==2.3.2](https://github.com/DLR-RM/stable-baselines3)
* [gymnasium==0.29.1](https://github.com/Farama-Foundation/Gymnasium)
* numpy, pandas, matplotlib
* PyTorch (CPU wheel; replace with CUDA wheel if you want GPU)

---

## 2. Configuration (JSON)

All runs are configured by a JSON file inside `conf/`.

### Example: `conf/input_parameters_20250911.json`

```json
{
  "case_name": "A034",
  "env_path_toactivate": "/home/USER/venv/bin",

  "size_actuator_array": 1,
  "size_obs_array_per_UDP": 4,
  "total_descarte": 1,
  "total_descarte_used": 1,

  "message_type": 1,
  "scalar_reward": 0.02,
  "reward_type": "CTA",

  "online_training": false,
  "online_inference": true,
  "offloading_training": false,
  "offload_inference": false,

  "evaluation": false,

  "eval_freq": 400,
  "n_eval_episodes": 2,
  "episode_length": 50,
  "total_episodes": 10000,

  "model_path": "/path/to/model_PPO_xxx.zip",
  "log_dir_template": "logs_v1_YYYYMMDD/model_PPO_{}",

  "crio_ip": "172.22.11.2",
  "hp_ip": "172.22.11.1",
  "debug_ip": "127.0.0.1",
  "udp_port_send": 61557,
  "udp_port_recv": 61555,

  "DEBUG": false,
  "debugging_IP": false,

  "action_min": -1,
  "action_max": 1,

  "n_steps": 100,
  "batch_size": 40,
  "n_epochs": 5,

  "algo_type": "PPO",
  "ppo_learning_rate": 0.02,
  "ppo_log_std_init": 0.7,
  "ppo_gamma": 0.5,
  "ppo_normalize_advantage": true,

  "actor_layers": [8, 8],
  "critic_layers": [8, 8],

  "inference_episodes": 1000,
  "inference_deterministic": true,
  "inference_print_every": 10,

  "trajectory_timeout": 5.0,
  "epochs_per_episode": 5,
  "identifier_str": "Control_id_x"
}
```

### Mode flags

Set **exactly one** of the following to `true`:

* `"online_training"`
* `"online_inference"`
* `"offloading_training"`
* `"offload_inference"`

The `make check-json JSON=...` command ensures your JSON is valid (keys present, port ranges OK, model exists if required, etc.).

---

## 3. Running

All runs go through `run_training_UDP_debug_v1_20250911.py`.
Use the **Makefile** to simplify commands.

### Online Training (SB3 PPO in Python)

```bash
make train-online JSON=input_parameters_20250911.json
```

### Online Inference (PPO acts, no training)

```bash
make infer-online JSON=input_parameters_20250911.json
```

### Offloading Training (CRIO executes, Python trains & resends weights)

```bash
make train-offload JSON=input_parameters_20250911.json
```

### Offload Inference (CRIO executes, Python just relays weights)

```bash
make infer-offload JSON=input_parameters_20250911.json
```

### Validate JSON mode & keys

```bash
make check-json JSON=input_parameters_20250911.json
```

---

## 4. Logs & Outputs

Each run creates a `LOG_DIR` based on the template in JSON:

```
logs_v1_YYYYMMDD/model_PPO_HHMM
```

### Files produced

| File                                | When              | Meaning                                                                   |
| ----------------------------------- | ----------------- | ------------------------------------------------------------------------- |
| `live_rewards.csv`                  | online modes      | Step log: `step,reward,action,timestamp,obs[-4],obs[-3],obs[-2],obs[-1]`. |
| `csv_log/live_rewards_temp.csv`     | all modes         | Rolling copy for live plotting.                                           |
| `env_monitor.*.csv`                 | online training   | SB3 Monitor log of episodes.                                              |
| `best_model.zip`, `evaluations.npz` | online training   | From EvalCallback (best checkpoint + eval history).                       |
| `model_PPO_<HHMMSS>.zip`            | online training   | Snapshots saved after each learning chunk.                                |
| `reward_vs_steps.png`               | if Monitor exists | Quick matplotlib plot of reward vs. timesteps.                            |
| `external_training.csv`             | offloading modes  | Per-episode log: `episode,steps,return,loss`.                             |
| `external_actor.pt`                 | offloading modes  | PyTorch checkpoint of offloaded actor.                                    |

### Utilities

```bash
make show-latest-log    # print newest logs* directory
make tail-live          # follow ./csv_log/live_rewards_temp.csv
make clean-tmp          # remove rolling temp CSV
```

---

## 5. Live Plotting

Use the provided `plot_live.py` script.

### Continuous live plot

```bash
make plot JSON=input_parameters_20250911.json
```

Watches `./csv_log/live_rewards_temp.csv` and updates every second.
Saves a PNG in `./figs/last_reward_plot_debug.png`.

### Plot a saved CSV

```bash
make plot-file JSON=input_parameters_20250911.json CSV=live_rewards.csv
```

Plots `./csv_log/live_rewards.csv`.

---

## 6. Local Simulators

Simulators let you test the workflow without hardware.

```bash
make sim-online_UDP JSON=input_parameters_20250911.json
make sim-offload_UDP JSON=input_parameters_20250911.json
```

Run agent + simulator together:

```bash
make debug-local-online JSON=input_parameters_20250911.json
make debug-local-offloading JSON=input_parameters_20250911.json
```

---

## 7. UDP Protocols

### Online modes (Python acts)

* **Action sent (type=1):**

  ```
  <timestamp>;1;1;1;1;1;1;A0;A1;...;Ak
  ```
* **Observation received:**

  ```
  <timestamp>;x1;x2;x3;x4
  ```

  repeated `TOTAL_DESCARTE+1` times → concatenated into one obs vector.

### Offloading modes (CRIO acts)

* **Weights (Python → CRIO):**

  ```
  # arch; weights; identifier
  OBS_H1_H2_..._ACTS;w1;w2;...;wN;IDENTIFIER
  ```
* **Trajectory (CRIO → Python):**

  ```
  s1,s2,...;a1,a2,...;r
  ...
  <END>
  ```

---

## 8. Quick Tips

* Always run `make check-json JSON=...` before launching — ensures config is valid.
* For dummy tests (no UDP), set `"evaluation": true` → env.reset() returns ones instead of waiting for UDP.
* `plot_live.py` is safe against partial writes (reads CSV line-buffered).
* If weight strings get too large for a single datagram, chunking will be needed (current version assumes fits in one UDP packet).



