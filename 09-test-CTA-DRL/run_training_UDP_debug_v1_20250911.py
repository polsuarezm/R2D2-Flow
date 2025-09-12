# === DRL-EXPERIMENTAL KV260
# === Full Training/Inference Script with Enhanced TensorBoard Logging, EvalCallback,
# === and CRIO Offloading Modes (trajectory-in / weights-out over UDP) ===

import socket, json, os, glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList

# Torch (used for offloading modes)
import torch
import torch.nn as nn
import torch.optim as optim


# ----------------- CLI -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--json_file", required=True)
args = parser.parse_args()
print(f"running case: {args.json_file}")


# ----------------- TB Callback -----------------
class TensorboardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        base_env = self.training_env.envs[0].env  # unwrap Monitor
        self.logger.record("custom/reward", base_env.last_reward)
        self.logger.record("custom/action", base_env.last_action)
        self.logger.record("custom/step_count", base_env.step_count)
        return True


# ----------------- Load configuration -----------------
with open(f"./conf/{args.json_file}", "r") as f:
    PARAMS = json.load(f)

DEBUG      = PARAMS.get("DEBUG", False)
DEBUG_IP   = PARAMS.get("debugging_IP", False)
ALGO_TYPE  = PARAMS.get("algo_type", "PPO").upper()

# NEW: Four explicit mode flags (mutually exclusive)
ONLINE_TRAIN   = bool(PARAMS.get("online_training", False))
ONLINE_INFER   = bool(PARAMS.get("online_inference", False))
OFFLOAD_TRAIN  = bool(PARAMS.get("offloading_training", False))
OFFLOAD_INFER  = bool(PARAMS.get("offload_inference", False))

# Optional: legacy evaluation flag only affects env.reset() (dummy ones() vs UDP)
EVAL_MODE = bool(PARAMS.get("evaluation", False))  # does NOT pick mode

# Guard: exactly one mode must be true
true_flags = [ONLINE_TRAIN, ONLINE_INFER, OFFLOAD_TRAIN, OFFLOAD_INFER]
if sum(true_flags) != 1:
    raise ValueError(
        "Exactly one of the following flags must be true: "
        "online_training, online_inference, offloading_training, offload_inference."
    )

# Model path (used for ONLINE_INFER; can be '.../model.zip' or base path without .zip)
_model_path = str(PARAMS.get("model_path", "")).strip()
MODEL_ZIP_PATH = _model_path if _model_path.endswith(".zip") else (_model_path + ".zip" if _model_path else "")

LOG_DIR = PARAMS["log_dir_template"].format(datetime.now().strftime("%Y%m%d-%H%M"))
os.makedirs(LOG_DIR, exist_ok=True)

# Save a copy of the JSON config inside the log directory
try:
    json_copy_path = os.path.join(LOG_DIR, "config.json")
    shutil.copyfile(f"./conf/{args.json_file}", json_copy_path)
    print(f"[INFO] Copied config to {json_copy_path}")
except Exception as e:
    print(f"[WARN] Could not copy config JSON: {e}")

ACTION_MIN = float(PARAMS["action_min"])
ACTION_MAX = float(PARAMS["action_max"])
N_STEPS    = int(PARAMS["n_steps"])
BATCH_SIZE = int(PARAMS["batch_size"])
N_EPOCHS   = int(PARAMS["n_epochs"])
N_OBS_ARRAY_PER_UDP = int(PARAMS["size_obs_array_per_UDP"])
N_ACTUATOR_ARRAY    = int(PARAMS["size_actuator_array"])

MESSAGE_TYPE = int(PARAMS["message_type"])
SCALAR_REW   = float(PARAMS["scalar_reward"])
TOTAL_DESCARTE       = int(PARAMS["total_descarte"])
TOTAL_DESCARTE_USED  = int(PARAMS["total_descarte_used"])
N_OBS_ARRAY = N_OBS_ARRAY_PER_UDP * (TOTAL_DESCARTE_USED + 1)

REWARD_TYPE = PARAMS.get("reward_type", "").upper()

# Inference controls (for ONLINE_INFER)
INFER_EPISODES       = int(PARAMS.get("inference_episodes", 1))
INFER_DETERMINISTIC  = bool(PARAMS.get("inference_deterministic", True))
INFER_PRINT_EVERY    = int(PARAMS.get("inference_print_every", 10))

# Offloading controls
TRAJ_TIMEOUT_S    = float(PARAMS.get("trajectory_timeout", 5.0))
EPOCHS_PER_EP     = int(PARAMS.get("epochs_per_episode", 5))
IDENTIFIER_STR    = PARAMS.get("identifier_str", "Control_id_x")


# ----------------- UDP setup -----------------
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.setblocking(False)

recv_ip = PARAMS["debug_ip"] if DEBUG_IP else PARAMS["hp_ip"]
send_ip = PARAMS["debug_ip"] if DEBUG_IP else PARAMS["crio_ip"]

sock_recv.bind((recv_ip, PARAMS["udp_port_recv"]))
print(f"Listening on {recv_ip}:{PARAMS['udp_port_recv']}")


# ----------------- Helpers for Offload Modes -----------------
def serialize_weights_like_keras_torch(actor: nn.Module, arch_header: str, identifier: str) -> str:
    """
    EXACT legacy message format:
      "# arch; weights; identifier\\n"
      "{arch_header};w1;w2;...;wN;{identifier}"
    Order: for each nn.Linear in build order -> weights (row-major), then bias.
    Returns the string (send as single UDP datagram).
    """
    flat_vals = []
    with torch.no_grad():
        for m in actor.modules():
            if isinstance(m, nn.Linear):
                flat_vals.append(m.weight.contiguous().view(-1).cpu().numpy())
                if m.bias is not None:
                    flat_vals.append(m.bias.contiguous().view(-1).cpu().numpy())
    flat = np.concatenate(flat_vals) if flat_vals else np.array([], dtype=np.float32)
    header = "# arch; weights; identifier\n"
    body   = arch_header + ";" + ";".join(f"{v:.5E}" for v in flat) + ";" + identifier
    return header + body


def recv_trajectory(sock_recv, episode_len, obs_dim, n_actions, timeout_s=5.0):
    """
    Receive one episode:
      packet: "s1,s2,...;a1,a2,...;r"
      terminator: "<END>"
    Returns (X, Y, R) with lengths <= episode_len.
    """
    sock_recv.settimeout(timeout_s)
    X  = np.zeros((episode_len, obs_dim), dtype=np.float32)
    Y  = np.zeros((episode_len, n_actions), dtype=np.float32)
    R  = np.zeros((episode_len, 1), dtype=np.float32)
    t  = 0
    try:
        while t < episode_len:
            data, _ = sock_recv.recvfrom(65507)  # max UDP datagram size
            if data == b"<END>":
                break
            decoded = data.decode().strip()
            st_str, at_str, rt_str = decoded.split(";")
            st = np.array([float(x) for x in st_str.split(",")], dtype=np.float32)
            at = np.array([float(x) for x in at_str.split(",")], dtype=np.float32)
            rt = float(rt_str)
            if st.size != obs_dim or at.size != n_actions:
                continue
            X[t] = st
            Y[t] = at
            R[t, 0] = rt
            t += 1
    except socket.timeout:
        pass
    finally:
        sock_recv.settimeout(None)
    return X[:t], Y[:t], R[:t]


class ExternalActor(nn.Module):
    """
    Simple MLP actor (Linear-ReLU...-Linear). Final layer is linear (no tanh),
    matching your earlier Keras example (activation=None).
    """
    def __init__(self, obs_dim, hidden, n_actions):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, n_actions)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------- Offloading Modes -----------------
def run_offloading(mode_train: bool):
    """
    mode_train=True  -> OFFLOADING_TRAIN (train from trajectories, resend weights)
    mode_train=False -> OFFLOAD_INFER (send weights, just log trajectories; no update)
    """
    tag = "OFFLOAD_TRAIN" if mode_train else "OFFLOAD_INFER"
    print(f"[{tag}] enabled (CRIO executes the policy).")

    # Dimensions (defaults infer from env sizing)
    obs_dim   = int(PARAMS.get("obs_dim", N_OBS_ARRAY))
    n_actions = int(PARAMS.get("n_actions", N_ACTUATOR_ARRAY))
    hidden    = PARAMS.get("actor_layers", [8, 8])

    # Build actor & optimizer
    actor = ExternalActor(obs_dim, hidden, n_actions)
    optimizer = optim.Adam(actor.parameters(), lr=float(PARAMS.get("ppo_learning_rate", 1e-3)))

    # Optional resume from last torch checkpoint
    ckpt_path = os.path.join(LOG_DIR, "external_actor.pt")
    if os.path.exists(ckpt_path):
        actor.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print(f"[{tag}] Loaded actor from {ckpt_path}")

    target_address = (send_ip, PARAMS["udp_port_send"])
    arch_header = f"{obs_dim}_" + "_".join(map(str, hidden)) + f"_{n_actions}"

    # Send initial weights (single datagram)
    msg = serialize_weights_like_keras_torch(actor, arch_header, IDENTIFIER_STR)
    sock_send.sendto(msg.encode("utf-8"), target_address)
    print(f"[{tag}] Sent initial model weights to CRIO.")

    total_eps = int(PARAMS.get("total_episodes", 1000))
    max_len   = int(PARAMS.get("episode_length", 1000))

    for ep in range(total_eps):
        print(f"[{tag}] Awaiting trajectory for episode {ep+1} ...")
        X, Y, R = recv_trajectory(sock_recv, max_len, obs_dim, n_actions, timeout_s=TRAJ_TIMEOUT_S)
        steps = len(X)
        if steps == 0:
            print(f"[{tag}] No trajectory received (timeout). Resending last weights and continuing.")
            sock_send.sendto(msg.encode("utf-8"), target_address)
            continue

        ret = float(R.sum())
        last_loss = np.nan

        if mode_train:
            # Advantage-weighted MSE between predicted actions and CRIO actions
            adv = R.squeeze()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            X_t = torch.from_numpy(X)
            Y_t = torch.from_numpy(Y)
            W_t = torch.from_numpy(adv.astype(np.float32)).view(-1, 1)

            actor.train()
            for _ in range(EPOCHS_PER_EP):
                optimizer.zero_grad()
                pred = actor(X_t)  # linear output
                loss = torch.mean(((pred - Y_t) ** 2) * (W_t.abs() + 1e-3))
                loss.backward()
                optimizer.step()
                last_loss = float(loss.item())

            # Save & refresh weight string
            torch.save(actor.state_dict(), ckpt_path)
            msg = serialize_weights_like_keras_torch(actor, arch_header, IDENTIFIER_STR)

        # Re-send weights every episode (both train & infer offload)
        sock_send.sendto(msg.encode("utf-8"), target_address)

        # Minimal CSV log
        with open(os.path.join(LOG_DIR, "external_training.csv"), "a") as f:
            f.write(f"{ep+1},{steps},{ret:.6f},{(last_loss if mode_train else np.nan):.6f}\n")

        print(f"[{tag}] Episode {ep+1}: steps={steps}, return={ret:.4f}"
              + (f", loss={last_loss:.6f}" if mode_train else "")
              + " | weights sent.")

    sock_send.close()
    sock_recv.close()
    print(f"Execution complete ({tag}). Logs in: {LOG_DIR}")
    raise SystemExit(0)


# ----------------- Custom Gym Environment (online modes) -----------------
class CRIOUDPEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, (N_OBS_ARRAY,), dtype=np.float32)
        self.action_space      = spaces.Box(ACTION_MIN, ACTION_MAX, (N_ACTUATOR_ARRAY,), dtype=np.float32)
        self.timestamp   = 0
        self.step_count  = 0
        self.global_step = 0
        self.last_obs    = np.zeros(N_OBS_ARRAY, dtype=np.float32)
        self.last_reward = 0.0
        self.last_action = 0.0

        os.makedirs("./csv_log", exist_ok=True)
        if os.path.exists("./csv_log/live_rewards_temp.csv"):
            os.remove("./csv_log/live_rewards_temp.csv")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        obs = np.ones(N_OBS_ARRAY, dtype=np.float32) if EVAL_MODE else self._receive_observation()
        return obs, {}

    def step(self, action):
        raw_action = action if isinstance(action, (list, np.ndarray)) else [action]

        if MESSAGE_TYPE == 1:
            # Example header with 6 dummy flags; adapt to your CRIO if needed
            message = f"{self.timestamp};1;1;1;1;1;1;" + ';'.join(map(str, raw_action))
        else:
            message = f"{self.timestamp};{''.join(map(str, raw_action))}"

        sock_send.sendto(message.encode(), (send_ip, PARAMS["udp_port_send"]))
        obs = self._receive_observation()

        reward = self._compute_reward_peak_fft_v1(obs) if REWARD_TYPE == "CTA" \
                 else self._compute_reward_debug_internalUDP(obs)

        if self.step_count % 10 == 0:
            print(f"| rew = {reward:.4f} | action = {action} | step = {self.step_count}")

        action_val = float(action[0]) if isinstance(action, (list, np.ndarray)) else float(action)
        self.last_action = action_val
        self.last_reward = reward

        with open(os.path.join(LOG_DIR, "live_rewards.csv"), "a") as archive_file, \
             open("./csv_log/live_rewards_temp.csv", "a") as tmp_file:
            self.global_step += 1
            archive_file.write(f"{self.global_step},{reward},{action_val},{self.timestamp},{obs[-4]},{obs[-3]},{obs[-2]},{obs[-1]}\n")
            tmp_file.write(f"{self.global_step},{reward},{action_val},{self.timestamp},{obs[-4]},{obs[-3]},{obs[-2]},{obs[-1]}\n")

        self.step_count += 1
        terminated = self.step_count >= PARAMS["episode_length"]
        return obs, reward, terminated, False, {}

    def _receive_observation(self):
        # Flush backlog
        sock_recv.setblocking(False)
        while True:
            try:
                sock_recv.recvfrom(1024)
            except BlockingIOError:
                break
        sock_recv.setblocking(True)

        accumulated_parts = np.zeros(N_OBS_ARRAY_PER_UDP * (TOTAL_DESCARTE + 1), dtype=np.float32)
        parts = None
        for i in range(TOTAL_DESCARTE + 1):
            data, _ = sock_recv.recvfrom(1024)
            parts = data.decode().strip().split(";")
            # Assuming 4 floats after timestamp (adjust if your payload changes)
            aux_obs = np.array([float(x) for x in parts[1:5]], dtype=np.float32)
            start = i * N_OBS_ARRAY_PER_UDP
            accumulated_parts[start:start + N_OBS_ARRAY_PER_UDP] = aux_obs

        sock_recv.setblocking(False)
        self.timestamp = int(parts[0]) if parts else self.timestamp
        self.last_obs = accumulated_parts[:N_OBS_ARRAY]
        return self.last_obs

    def _compute_reward_peak_fft_v1(self, obs_pre_reward):
        aux = obs_pre_reward[-2]
        return 1 - aux / SCALAR_REW

    def _compute_reward_debug_internalUDP(self, obs_pre_reward):
        return float(obs_pre_reward[-1])

    def render(self, mode="human"): pass
    def close(self): pass


# ----------------- Env setup (online modes) -----------------
def make_env():
    env = CRIOUDPEnv()
    return Monitor(env, filename=os.path.join(LOG_DIR, "env_monitor"))

env = DummyVecEnv([make_env])


# ----------------- PPO builder (online training) -----------------
def build_ppo(env_):
    return PPO(
        "MlpPolicy", env_, verbose=1,
        learning_rate=PARAMS.get("ppo_learning_rate", 1e-3),
        device="cpu",
        n_steps=int(PARAMS["n_steps"]),
        batch_size=int(PARAMS["batch_size"]),
        n_epochs=int(PARAMS["n_epochs"]),
        gamma=PARAMS.get("ppo_gamma", 0.99),
        tensorboard_log=LOG_DIR,
        policy_kwargs=dict(
            net_arch=dict(
                pi=PARAMS.get("actor_layers", [8]),
                vf=PARAMS.get("critic_layers", [16, 64, 64])
            ),
            log_std_init=PARAMS.get("ppo_log_std_init", -0.5),
        ),
    )


# ----------------- MAIN BRANCHING -----------------
if OFFLOAD_TRAIN or OFFLOAD_INFER:
    # CRIO executes policy; Python sends/receives weights & trajectories
    run_offloading(mode_train=OFFLOAD_TRAIN)

elif ONLINE_INFER:
    # Load PPO and run inference only (Python acts; no training)
    if not MODEL_ZIP_PATH or not os.path.exists(MODEL_ZIP_PATH):
        raise FileNotFoundError(
            f"[INFERENCE] model_path not found: {MODEL_ZIP_PATH}. "
            f"Please set 'model_path' to a valid SB3 .zip."
        )

    print(f"[INFERENCE] Loading {ALGO_TYPE} model from: {MODEL_ZIP_PATH}")
    model = PPO.load(MODEL_ZIP_PATH, env=None, device="cpu")

    base_env = env.envs[0]  # Monitor-wrapped Gymnasium env

    print(f"[INFERENCE] Running {INFER_EPISODES} episode(s), deterministic={INFER_DETERMINISTIC}")
    for ep in range(INFER_EPISODES):
        obs, info = base_env.reset()
        ep_rew = 0.0
        steps = 0
        while True:
            action, _ = model.predict(obs, deterministic=INFER_DETERMINISTIC)
            obs, reward, terminated, truncated, info = base_env.step(action)
            ep_rew += float(reward)
            steps += 1
            if steps % INFER_PRINT_EVERY == 0:
                print(f"[INFER] ep {ep+1} step {steps} | rew={float(reward):.4f} | last_ts={base_env.env.timestamp}")
            if terminated or truncated:
                print(f"[INFER] Episode {ep+1} finished: steps={steps}, return={ep_rew:.4f}")
                break

else:
    # ONLINE_TRAIN = True
    print("[TRAIN] Online training mode (SB3 PPO).")
    eval_callback = EvalCallback(
        env,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        eval_freq=PARAMS.get("eval_freq", 5000),
        n_eval_episodes=PARAMS.get("n_eval_episodes", 1),
        deterministic=True,
        render=False
    )
    callback = CallbackList([TensorboardLoggingCallback(), eval_callback])

    if ALGO_TYPE != "PPO":
        raise NotImplementedError("Only PPO supported in this script version.")

    model = build_ppo(env)

    # Learn in chunks of N_STEPS; save a timestamped snapshot after each chunk
    total_chunks = int(PARAMS["total_episodes"] * PARAMS["episode_length"] // N_STEPS)
    for _ in range(total_chunks):
        model.learn(total_timesteps=N_STEPS, reset_num_timesteps=False, callback=callback)
        model.save(os.path.join(LOG_DIR, f"model_{ALGO_TYPE}_{datetime.now().strftime('%H%M%S')}"))


# ----------------- Training monitor plot (harmless during inference) -----------------
monitor_files = glob.glob(os.path.join(LOG_DIR, "*monitor.csv"))
if monitor_files:
    df = pd.read_csv(monitor_files[0], skiprows=1)
    if 'r' in df.columns and 't' in df.columns:
        plt.figure(figsize=(8, 4))
        plt.plot(df["t"], df["r"], marker='o', linestyle='None')
        plt.xlabel("Timesteps")
        plt.ylabel("Episode Reward")
        plt.title(f"Progress - {ALGO_TYPE}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(LOG_DIR, "reward_vs_steps.png"))

sock_send.close()
sock_recv.close()
print("Execution complete. Logs saved in:", LOG_DIR)

