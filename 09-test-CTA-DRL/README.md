Here’s a clean, copy-pasteable **README** for your repo. It explains setup, how to launch each mode, the JSON flags, and where every log/file lands.

---

# DRL-EXPERIMENTAL KV260 — UDP RL Framework

This repo runs a PPO-based RL loop over UDP to a CRIO/KV260 system.
It supports three execution modes:

1. **Training mode (SB3 PPO + EvalCallback)** – Python computes actions online over UDP.
2. **Inference-only mode** – load a saved PPO and act online; no training.
3. **Offloading-weights mode** – the device (CRIO) runs the policy; Python only receives full trajectories, updates the actor, and sends updated **weights** back via UDP (same message format you used before).

---

## 1) Requirements

* **Python** 3.9–3.11 (recommended)
* **pip** and a virtualenv/conda recommended

### Python packages

```bash
pip install \
  stable-baselines3==2.3.2 \
  gymnasium==0.29.1 \
  numpy \
  pandas \
  matplotlib \
  torch --index-url https://download.pytorch.org/whl/cpu
```

> If you have a GPU PyTorch, install the proper CUDA wheel instead of the CPU one.

---

## 2) Files & Structure

* `run.py` – your main script (the big one you just updated).
* `conf/<case>.json` – configuration file(s) (example below).
* **Generated at runtime (per run)** inside a fresh `LOG_DIR`:

  * `env_monitor.*.csv` – SB3 Monitor logs (episode stats).
  * `live_rewards.csv` – streaming log of step-wise reward/action/timestamp/4 last obs (always).
  * `./csv_log/live_rewards_temp.csv` – rolling temp copy for live plotting (overwritten each run).
  * `reward_vs_steps.png` – quick plot of episode rewards vs. time (if monitor log exists).
  * `best_model.zip` / `evaluations.npz` – from `EvalCallback` when training.
  * `model_PPO_<HHMMSS>.zip` – periodic model snapshots saved during training.
  * `external_training.csv` – only in **offloading-weights mode**; per-episode summary.
  * `external_actor.pt` – only in offloading mode; PyTorch actor checkpoint.

---

## 3) JSON Configuration (key flags)

Minimal example (adapt to your network & case):

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

  "training": false,
  "evaluation": true,

  "eval_freq": 400,
  "n_eval_episodes": 2,
  "episode_length": 50,
  "total_episodes": 10000,

  "create_new_model": false,
  "load_model_path": true,
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

  "offloading_weights_mode": false,
  "trajectory_timeout": 5.0,
  "epochs_per_episode": 5,
  "identifier_str": "Control_id_x"
}
```

### Semantics of key flags

* **Mode selection**

  * **Training** (SB3 PPO): happens when `offloading_weights_mode=false` **and** NOT in inference-only (see below).
  * **Inference-only**: when `create_new_model=false` **and** (`load_model_path==true` or `load_model_path` is a string path) — the script loads `model_path` and only predicts actions.
  * **Offloading-weights mode**: when `offloading_weights_mode=true` — CRIO executes the policy; Python only trains on received trajectories and sends weights back.

* **`evaluation`** (`EVAL_MODE`): affects only `env.reset()` behavior:

  * `true` → returns a dummy obs of ones at reset (no UDP read on reset). Useful for dry runs.
  * `false` → normal UDP receive on reset.

* **Networking**

  * Listener binds to `hp_ip:udp_port_recv`.
  * Actions are sent to `crio_ip:udp_port_send`.
  * Set `"debugging_IP": true` to switch both to `debug_ip`.

* **Dimensions**

  * Observation length used by env = `N_OBS_ARRAY = size_obs_array_per_UDP * (total_descarte_used + 1)`
  * Action dimension = `size_actuator_array`.

---

## 4) Running

Make sure your venv is active and the UDP peer is reachable.

```bash
python run.py --json_file A034.json
```

### a) Training mode (SB3 PPO)

Set in JSON:

```json
"offloading_weights_mode": false,
"create_new_model": true,
"load_model_path": false
```

or simply omit `model_path` and keep `create_new_model: true`.

* The script will:

  * build PPO,
  * learn in chunks of `n_steps`,
  * every `eval_freq` steps run `n_eval_episodes` evaluations and save `best_model.zip`,
  * periodically save `model_PPO_<HHMMSS>.zip`.

### b) Inference-only mode

Set in JSON:

```json
"offloading_weights_mode": false,
"create_new_model": false,
"load_model_path": true,
"model_path": "/path/to/model_PPO_xxx.zip",
"inference_episodes": 1000
```

* Loads the PPO and runs `inference_episodes` in a loop, sending actions online over UDP.

### c) Offloading-weights mode (CRIO drives the policy)

Set in JSON:

```json
"offloading_weights_mode": true
```

* Python:

  * Sends initial weights (single UDP message, **same format** as your legacy Keras script).
  * Waits for a full trajectory stream (packets `state_csv;action_csv;reward`; terminated by `<END>`).
  * Trains a small PyTorch actor using advantage-weighted MSE.
  * Saves and re-sends updated weights (same single-message format).
  * Logs to `external_training.csv` and `external_actor.pt`.

---

## 5) UDP Message Protocols

### 5.1 Online actions (training & inference modes)

* **To CRIO** each `step()`:

  * If `message_type == 1`:

    ```
    "<timestamp>;1;1;1;1;1;1;A0;A1;...;Ak"
    ```
  * Else:

    ```
    "<timestamp>;A0A1...Ak"   # compact form; you used this earlier
    ```

* **From CRIO** (observations):
  For each `reset/step`, the env collects `(total_descarte + 1)` UDP packets.
  Each packet:

  ```
  "<timestamp>;x1;x2;x3;x4"
  ```

  We keep the last `size_obs_array_per_UDP` (here 4) and pack them into the observation buffer.
  Reward is computed locally from the final obs chunk depending on `"reward_type"`:

  * `"CTA"` → `reward = 1 - obs[-2] / scalar_reward`
  * else → `reward = obs[-1]`

### 5.2 Offloading weights (offloading\_weights\_mode)

* **Weights → CRIO** (single datagram):

  ```
  # arch; weights; identifier
  OBS_DIM_H1_H2_..._Hn_ACTS; w1; w2; ...; wN; IDENTIFIER_STR
  ```

  * Weight order: for each Linear layer in build order → **weights (row-major) then bias**.
  * Matches your Keras string convention.

* **Trajectory → Python** (multiple datagrams, then terminator):

  ```
  s1,s2,...;a1,a2,...;r
  ...
  <END>
  ```

  * Expected sizes: `len(state)==obs_dim`, `len(action)==n_actions`.

---

## 6) Outputs & Where to Find Them

Inside `LOG_DIR = log_dir_template.format(YYYYMMDD-HHMM)`:

* `live_rewards.csv`
  Columns: `global_step,reward,action0,timestamp,obs[-4],obs[-3],obs[-2],obs[-1]`
  Written **every step** (all modes that use the Gym env).

* `./csv_log/live_rewards_temp.csv`
  Same columns as above; overwritten per run (handy for dashboards).

* `env_monitor.*.csv` (SB3 Monitor)
  Episode summaries while training (steps, reward, length). SB3 writes the header row comment (skiprows=1).

* `best_model.zip` & `evaluations.npz`
  From `EvalCallback` in **training mode**.

* `model_PPO_<HHMMSS>.zip`
  Periodic snapshots saved each `learn()` cycle.

* `reward_vs_steps.png`
  Quick scatter plot of episode reward vs elapsed timesteps (if a monitor file exists).

* `external_training.csv` (**offloading mode only**)
  CSV rows: `episode,steps,return,loss`

* `external_actor.pt` (**offloading mode only**)
  PyTorch checkpoint for the external actor.

---

## 7) Tips & Troubleshooting

* **Dummy obs vs. real UDP on reset**:
  `"evaluation": true` → `reset()` returns `ones(...)`.
  Set it to `false` for real UDP receive at reset.

* **VecEnv vs. raw env loops**:
  In inference we unwrap: `base_env = env.envs[0]` to use Gymnasium API:
  `obs, info = base_env.reset()` → `obs, reward, terminated, truncated, info = base_env.step(action)`.

* **Model path**:
  You can give either `".../model.zip"` or base path without `.zip`; the script handles both.

* **UDP backlog**:
  The env flushes the receive socket before each blocking read to avoid stale packets.

* **Ports and IPs**:
  Make sure firewall allows UDP both ways for `udp_port_send`/`udp_port_recv`.
  If running locally, set `"debugging_IP": true` to switch to `debug_ip` (127.0.0.1).

---

## 8) Repro Checklists

* **Training (online actions)**

  * `offloading_weights_mode=false`
  * `create_new_model=true` (or resume)
  * Proper `hp_ip`, `crio_ip`, ports open
  * Observe `best_model.zip`, periodic `model_PPO_*.zip`, `live_rewards.csv`

* **Inference**

  * `offloading_weights_mode=false`
  * `create_new_model=false`, `load_model_path=true`, `model_path="...zip"`
  * Watch `live_rewards.csv` and console prints

* **Offloading**

  * `offloading_weights_mode=true`
  * Device sends `state_csv;action_csv;reward` … `<END>`
  * Python logs `external_training.csv`, sends weight string each episode


