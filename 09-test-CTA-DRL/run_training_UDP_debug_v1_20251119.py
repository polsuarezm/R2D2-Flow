# === DRL-EXPERIMENTAL KV260
# === Full Training/Inference Script with Enhanced TensorBoard Logging, EvalCallback,
# === and CRIO Offloading Modes (trajectory-in / weights-out over UDP) ===

import socket, json, os, glob, shutil, time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import argparse
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.logger import HParam

# Torch (used for offloading modes)
import torch
import torch.nn as nn
import torch.optim as optim


# ----------------- CLI -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--json_file", required=True)
args = parser.parse_args()
print(f"running case: {args.json_file}")


# ----------------- Load configuration -----------------
with open(f"./{args.json_file}", "r") as f:
    PARAMS = json.load(f)

DEBUG      = PARAMS.get("DEBUG", False)
DEBUG_IP   = PARAMS.get("debugging_IP", False)
ALGO_TYPE  = PARAMS.get("algo_type", "PPO").upper()

# Four explicit mode flags (mutually exclusive)
ONLINE_TRAIN   = bool(PARAMS.get("online_training", False))
ONLINE_INFER   = bool(PARAMS.get("online_inference", False))
OFFLOAD_TRAIN  = bool(PARAMS.get("offloading_training", False))
OFFLOAD_INFER  = bool(PARAMS.get("offload_inference", False))

# Legacy evaluation flag only affects env.reset() (dummy ones() vs UDP)
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
    shutil.copyfile(f"./{args.json_file}", json_copy_path)
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
SCALAR_REW_FFT   = float(PARAMS["scalar_reward_fft"])
SCALAR_REW_MEANU = float(PARAMS["scalar_reward_meanu"])
ALPHA_FFT    = float(PARAMS["alpha_fft"])
BETA_MEAN    = float(PARAMS["beta_meanU"])
RE_D         = int(PARAMS["re_d"])
SKIP_FIRST_UDP       = int(PARAMS["skip_first_udp"])
SAMPLE_HISTORY_SIZE  = int(PARAMS["sample_history_size"])  # how many UDPs to accumulate (in addition to current)
N_OBS_ARRAY = N_OBS_ARRAY_PER_UDP * (SAMPLE_HISTORY_SIZE + 1)

time.sleep(1.0)  # brief pause to ensure log dir is ready

REWARD_TYPE = PARAMS.get("reward_type", "").upper()

# Inference controls (for ONLINE_INFER)
INFER_EPISODES       = int(PARAMS.get("inference_episodes", 1))
INFER_DETERMINISTIC  = bool(PARAMS.get("inference_deterministic", True))
INFER_PRINT_EVERY    = int(PARAMS.get("inference_print_every", 10))

# Offloading controlsTRAJ_TIMEOUT_S    = float(PARAMS.get("trajectory_timeout", 5.0))
TRAJ_TIMEOUT_S    = float(PARAMS.get("trajectory_timeout", 20.0))
IDENTIFIER_STR    = PARAMS.get("identifier_str", "Control_id_2")


# ----------------- UDP setup -----------------
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.setblocking(False)

recv_ip = PARAMS["debug_ip"] if DEBUG_IP else PARAMS["hp_ip"]
send_ip = PARAMS["debug_ip"] if DEBUG_IP else PARAMS["crio_ip"]

sock_recv.bind((recv_ip, PARAMS["udp_port_recv"]))
print(f"Listening on {recv_ip}:{PARAMS['udp_port_recv']}")


# ----------------- Enhanced TensorBoard callback -----------------
class EnhancedTensorboardLoggingCallback(BaseCallback):
    """
    Custom Stable-Baselines3 callback that logs rich per-step and per-episode
    diagnostics to TensorBoard (obs/action stats, reward stats, timing, etc.).
    """

    def __init__(self, rolling=500, verbose=0):
        """
        Initialize the callback.

        Parameters
        ----------
        rolling : int
            Window size for rolling averages (reward, action, obs).
        verbose : int
            Verbosity level for BaseCallback.
        """
        super().__init__(verbose)
        self.rolling = rolling
        self.rew_buf = deque(maxlen=rolling)
        self.act_buf = deque(maxlen=rolling)
        self.obs_buf = deque(maxlen=rolling)
        self._last_wall_t = None
        self._last_ts = None
        self._logged_hparams = False
        self._ep_ret = 0.0
        self._ep_len = 0

    def _on_training_start(self) -> None:
        """Record hyperparameters to TensorBoard once at training start."""
        if not self._logged_hparams:
            try:
                hparams = dict(
                    algo=ALGO_TYPE,
                    ppo_lr=PARAMS.get("ppo_learning_rate", 1e-3),
                    ppo_gamma=PARAMS.get("ppo_gamma", 0.99),
                    n_steps=N_STEPS,
                    batch_size=BATCH_SIZE,
                    n_epochs=N_EPOCHS,
                    actor_layers="x".join(map(str, PARAMS.get("actor_layers", [8]))),
                    critic_layers="x".join(map(str, PARAMS.get("critic_layers", [16,64,64]))),
                    obs_chunk=N_OBS_ARRAY_PER_UDP,
                    history=SAMPLE_HISTORY_SIZE,
                    skip_first_udp=SKIP_FIRST_UDP,
                    action_min=ACTION_MIN,
                    action_max=ACTION_MAX,
                    reward_type=REWARD_TYPE,
                )
                self.logger.record("hparams", HParam(hparams, {}))
                self._logged_hparams = True
            except Exception:
                pass

    def _on_step(self) -> bool:
        """Log per-step statistics (obs, actions, rewards, timing, etc.) to TensorBoard."""
        # unwrap to the base env (Monitor -> CRIOUDPEnv)
        base_env = self.training_env.envs[0].env
        obs = base_env.last_obs
        rew = float(base_env.last_reward)
        act = float(base_env.last_action)

        # rolling buffers
        self.rew_buf.append(rew)
        self.act_buf.append(act)
        self.obs_buf.append(obs if isinstance(obs, np.ndarray) else np.asarray(obs, dtype=np.float32))

        # timing
        now = time.time()
        steps_per_sec = None
        if self._last_wall_t is not None:
            dt = max(1e-9, now - self._last_wall_t)
            steps_per_sec = 1.0 / dt
        self._last_wall_t = now

        # UDP timestamp delta (if increasing)
        udp_dt = None
        if self._last_ts is not None:
            udp_dt = base_env.timestamp - self._last_ts
        self._last_ts = base_env.timestamp

        # episode accounting (Monitor also logs CSV, but we mirror to TB)
        self._ep_ret += rew
        self._ep_len += 1
        # If episode just terminated, Monitor would have reset; we can detect via step_count
        if base_env.step_count == 0 and self.num_timesteps > 0:
            # episode just reset in env.reset() after a done in previous step
            self.logger.record("custom/episode_return", self._ep_ret)
            self.logger.record("custom/episode_length", self._ep_len)
            self._ep_ret, self._ep_len = 0.0, 0

        # obs stats
        if isinstance(obs, np.ndarray):
            self.logger.record("obs/min", float(np.min(obs)))
            self.logger.record("obs/max", float(np.max(obs)))
            self.logger.record("obs/mean", float(np.mean(obs)))
            self.logger.record("obs/std", float(np.std(obs)))
            # tail channels (your CSV dumps last 4)
            if obs.shape[0] >= 4:
                self.logger.record("obs/tail_c0", float(obs[-4]))
                self.logger.record("obs/tail_c1", float(obs[-3]))
                self.logger.record("obs/tail_c2", float(obs[-2]))
                self.logger.record("obs/tail_c3", float(obs[-1]))

        # action + reward stats
        self.logger.record("action/value", act)
        self.logger.record("reward/value", rew)
        if len(self.rew_buf) >= 2:
            self.logger.record("reward/rolling_mean", float(np.mean(self.rew_buf)))
            self.logger.record("reward/rolling_std", float(np.std(self.rew_buf)))
        if len(self.act_buf) >= 2:
            self.logger.record("action/rolling_mean", float(np.mean(self.act_buf)))
            self.logger.record("action/rolling_std", float(np.std(self.act_buf)))

        # loop timing
        if steps_per_sec is not None:
            self.logger.record("timing/steps_per_sec", float(steps_per_sec))
        if udp_dt is not None:
            self.logger.record("timing/udp_dt", float(udp_dt))

        # PPO learning rate (visible in TB)
        try:
            # SB3's lr_schedule expects current progress remaining or num_timesteps depending version.
            lr = self.model.lr_schedule(1.0) if callable(self.model.lr_schedule) else None
            if lr is None or isinstance(lr, (list, tuple)):
                lr = self.model.learning_rate if hasattr(self.model, "learning_rate") else None
            if lr is not None:
                self.logger.record("train/learning_rate_cb", float(lr if np.isscalar(lr) else lr[0]))
        except Exception:
            pass

        # Also keep your original custom signals
        self.logger.record("custom/reward", base_env.last_reward)
        self.logger.record("custom/action", base_env.last_action)
        self.logger.record("custom/step_count", base_env.step_count)

        return True


# ----------------- Helpers for Offload Modes -----------------
def serialize_weights_like_keras_torch(actor: nn.Module, arch_header: str, identifier: str) -> str:
    """
    Flatten a PyTorch actor network's Linear weights/biases into a single
    semicolon-separated string, prefixed with a simple architecture header
    and suffixed with an identifier string.

    Parameters
    ----------
    actor : nn.Module
        PyTorch actor network whose parameters will be serialized.
    arch_header : str
        Simple descriptor of network architecture (e.g. "obs_h1_h2_act").
    identifier : str
        Additional identifier appended at the end of the message.

    Returns
    -------
    str
        Encoded string "arch_header;w1;w2;...;identifier".
    """
    flat_vals = []
    with torch.no_grad():
        for m in actor.modules():
            if isinstance(m, nn.Linear):
                flat_vals.append(m.weight.contiguous().view(-1).cpu().numpy())
                if m.bias is not None:
                    flat_vals.append(m.bias.contiguous().view(-1).cpu().numpy())
    flat = np.concatenate(flat_vals) if flat_vals else np.array([], dtype=np.float32)
    body   = arch_header + ";" + ";".join(f"{v:.5E}" for v in flat) + ";" + identifier
    return body


def compute_reward_from_obs(obs_batch: np.ndarray) -> np.ndarray:
    """
    Compute per-step rewards from a batch of full observations using the CTA_3
    definition, matching CRIOUDPEnv._compute_reward_peak_fft_v3.

    Parameters
    ----------
    obs_batch : np.ndarray
        Array of observations with shape (T, obs_dim_full).

    Returns
    -------
    np.ndarray
        Rewards with shape (T, 1). If REWARD_TYPE != "CTA_3", returns zeros.
    """
    if REWARD_TYPE != "CTA_3":
        # Fallback: reward = 0 if a different reward is configured.
        return np.zeros((obs_batch.shape[0], 1), dtype=np.float32)

    # In the online env:
    #   aux_fftpeak = obs[-2]
    #   aux_meanU   = obs[-4]
    aux_fftpeak = obs_batch[:, -2]   # (T,)
    aux_meanU   = obs_batch[:, -4]   # (T,)

    mean_term = 1.5 - aux_meanU
    rewards = (
        ALPHA_FFT * (1.0 - aux_fftpeak / SCALAR_REW_FFT)
        + BETA_MEAN * (mean_term / SCALAR_REW_MEANU)
    )

    return rewards.reshape(-1, 1).astype(np.float32)


def recv_trajectory(sock_recv, episode_len, obs_dim, n_actions):
    """
    Minimal version for debugging.

    Assumptions:
      - Exactly ONE UDP datagram contains the FULL episode:
            obs1[0:obs_dim]; act1[0:n_actions];
            obs2[0:obs_dim]; act2[0:n_actions];
            ...
      - No timeouts, no try/except: will block on recvfrom()
        until some packet arrives or you Ctrl+C.

    Returns:
      X : (T, obs_dim)
      Y : (T, n_actions)
      R : (T, 1)  via compute_reward_from_obs(X)
    """

    per_step_tokens = obs_dim + n_actions
    if per_step_tokens <= 0:
        raise ValueError(
            f"[recv_trajectory] Invalid per_step_tokens={per_step_tokens} "
            f"(obs_dim={obs_dim}, n_actions={n_actions})"
        )

    # Blocking receive: waits here until *one* datagram arrives
    print("[recv_trajectory] Waiting for one UDP datagram with full episode...")
    sock_recv.setblocking(True)
    data, addr = sock_recv.recvfrom(65507)
    print(f"[recv_trajectory] Received datagram from {addr}, bytes={len(data)}")

    decoded = data.decode().strip()
    tokens = [t for t in decoded.split(";") if t != ""]
    print(f"[recv_trajectory] Total tokens in datagram: {len(tokens)}")

    if len(tokens) % per_step_tokens != 0:
        raise RuntimeError(
            f"[recv_trajectory] Token count {len(tokens)} is not a multiple of "
            f"per_step_tokens={per_step_tokens} (obs_dim={obs_dim}, n_actions={n_actions})."
        )

    max_steps = len(tokens) // per_step_tokens
    if max_steps == 0:
        raise RuntimeError(
            f"[recv_trajectory] Buffer too short: {len(tokens)} tokens, "
            f"need at least {per_step_tokens} for one step."
        )

    flat_arr = np.array([float(t) for t in tokens], dtype=np.float32)
    traj_mat = flat_arr.reshape(T, per_step_tokens)  # (T, obs_dim + n_actions)

    # Split into obs and actions
    X = traj_mat[:, :obs_dim].astype(np.float32)
    if n_actions > 0:
        Y = traj_mat[:, obs_dim:obs_dim + n_actions].astype(np.float32)
    else:
        Y = np.zeros((T, 0), dtype=np.float32)

    # Compute rewards from obs (CTA_3 etc.)
    R = compute_reward_from_obs(X)  # must return (T, 1)

    print(f"[recv_trajectory] Got trajectory: steps={T}, obs_dim={obs_dim}, n_actions={n_actions}")
    return X, Y, R


class ExternalActor(nn.Module):
    """
    Simple feed-forward actor network used in offloading modes.
    The CRIO device runs a compatible network; this class is used
    to train and serialize weights on the host.
    """

    def __init__(self, obs_dim, hidden, n_actions):
        """
        Build a fully-connected MLP actor.

        Parameters
        ----------
        obs_dim : int
            Input dimension (history-augmented observation).
        hidden : list of int
            Sizes of hidden layers.
        n_actions : int
            Number of continuous actions.
        """
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, n_actions)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass mapping observations to actions.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of observations with shape (B, obs_dim).

        Returns
        -------
        torch.Tensor
            Output batch of actions with shape (B, n_actions).
        """
        return self.net(x)


# ----------------- Offloading Modes -----------------
def run_offloading(mode_train: bool):
    """
    Execute the offloading pipeline where the CRIO executes the policy
    (online) and sends complete trajectories to the host for offline
    weight updates.

    Parameters
    ----------
    mode_train : bool
        If True, perform training on received trajectories.
        If False, only receive trajectories and resend weights (no updates).
    """
    tag = "OFFLOAD_TRAIN" if mode_train else "OFFLOAD_INFER"
    print(f"[{tag}] enabled (CRIO executes the policy).")

    obs_dim   = int(PARAMS.get("obs_dim", N_OBS_ARRAY))
    n_actions = int(PARAMS.get("n_actions", N_ACTUATOR_ARRAY))
    hidden    = PARAMS.get("actor_layers", [8, 8])

    actor = ExternalActor(obs_dim, hidden, n_actions)
    optimizer = optim.Adam(actor.parameters(), lr=float(PARAMS.get("ppo_learning_rate", 1e-3)))

    ckpt_path = os.path.join(LOG_DIR, "external_actor.pt")
    if os.path.exists(ckpt_path):
        actor.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print(f"[{tag}] Loaded actor from {ckpt_path}")

    target_address = (send_ip, PARAMS["udp_port_send"])
    arch_header = f"{obs_dim}_" + "_".join(map(str, hidden)) + f"_{n_actions}"

    msg = serialize_weights_like_keras_torch(actor, arch_header, IDENTIFIER_STR)
    print("POOOOL --> ", msg)
    sock_send.sendto(msg.encode("utf-8"), target_address)
    print(f"[{tag}] Sent initial model weights to CRIO.")

    total_eps = int(PARAMS.get("total_episodes", 1000))
    max_len   = int(PARAMS.get("episode_length", 1000))

    for ep in range(total_eps):
        print(f"[{tag}] Awaiting trajectory for episode {ep+1} ...")
        X, Y, R = recv_trajectory(sock_recv, max_len, obs_dim, n_actions)
        steps = len(X)
        if steps == 0:
            print(f"[{tag}] No trajectory received (timeout). Resending last weights and continuing.")
            sock_send.sendto(msg.encode("utf-8"), target_address)
            continue

        ret = float(R.sum())
        last_loss = np.nan

        if mode_train:
            # Simple weighted MSE loss with advantage-like weights from R
            adv = R.squeeze()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            X_t = torch.from_numpy(X)
            Y_t = torch.from_numpy(Y)
            W_t = torch.from_numpy(adv.astype(np.float32)).view(-1, 1)

            actor.train()
            for _ in range(int(PARAMS.get("epochs_per_episode", 5))):
                optimizer.zero_grad()
                pred = actor(X_t)
                loss = torch.mean(((pred - Y_t) ** 2) * (W_t.abs() + 1e-3))
                loss.backward()
                optimizer.step()
                last_loss = float(loss.item())

            torch.save(actor.state_dict(), ckpt_path)
            msg = serialize_weights_like_keras_torch(actor, arch_header, IDENTIFIER_STR)

        sock_send.sendto(msg.encode("utf-8"), target_address)

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
    """
    Gymnasium environment that wraps the CRIO UDP-based hardware in the loop.
    Handles action sending, observation receiving with history, and reward
    computation for online modes (training/inference).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        """
        Initialize the CRIO UDP environment:
            - define observation/action spaces
            - set up internal buffers and counters
            - prepare CSV logging for live rewards
        """
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, (N_OBS_ARRAY,), dtype=np.float32)
        self.action_space      = spaces.Box(ACTION_MIN, ACTION_MAX, (N_ACTUATOR_ARRAY,), dtype=np.float32)

        self.timestamp   = 0
        self.step_count  = 0
        self.global_step = 0
        self.last_obs    = np.zeros(N_OBS_ARRAY, dtype=np.float32)
        self.last_reward = 0.0
        self.last_action = 0.0

        # === Persistent flow-history buffer (oldest → newest) ===
        self.history_len = SAMPLE_HISTORY_SIZE + 1
        self.chunk_size  = N_OBS_ARRAY_PER_UDP
        self._hist = deque(
            [np.zeros(self.chunk_size, dtype=np.float32) for _ in range(self.history_len)],
            maxlen=self.history_len
        )

        os.makedirs("./csv_log", exist_ok=True)
        if os.path.exists("./csv_log/live_rewards_temp.csv"):
            os.remove("./csv_log/live_rewards_temp.csv")

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment at the start of an episode.

        If EVAL_MODE is True: return a ones-vector observation.
        Otherwise: receive a fresh observation from UDP.
        """
        super().reset(seed=seed)
        self.step_count = 0
        obs = np.ones(N_OBS_ARRAY, dtype=np.float32) if EVAL_MODE else self._receive_observation()
        return obs, {}

    def step(self, action):
        """
        Apply an action to the CRIO system via UDP and advance one timestep.

        Parameters
        ----------
        action : np.ndarray or list
            Continuous action(s) to send.

        Returns
        -------
        obs : np.ndarray
            New observation after applying the action (with history).
        reward : float
            Scalar reward for this transition.
        terminated : bool
            Whether the episode has terminated by length.
        truncated : bool
            Unused (False).
        info : dict
            Extra information (empty).
        """
        raw_action = action if isinstance(action, (list, np.ndarray)) else [action]

        if MESSAGE_TYPE == 1:
            message = f"{self.timestamp};1;1;1;1;1;1;" + ';'.join(map(str, raw_action))
        else:
            message = f"{self.timestamp};" + ';'.join(map(str, raw_action))

        sock_send.sendto(message.encode(), (send_ip, PARAMS["udp_port_send"]))
        obs = self._receive_observation()

        if REWARD_TYPE == "CTA_1":
            reward = self._compute_reward_peak_fft_v1(obs)
        elif REWARD_TYPE == "CTA_2":
            reward = self._compute_reward_peak_fft_v2(obs, RE_D, ALPHA_FFT, BETA_MEAN)
        elif REWARD_TYPE == "CTA_3":
            reward = self._compute_reward_peak_fft_v3(obs, RE_D, ALPHA_FFT, BETA_MEAN)
        else:
            reward = self._compute_reward_debug_internalUDP(obs)

        if self.step_count % 10 == 0:
            print(f"| rew = {reward:.4f} | action = {action} | step = {self.step_count}")

        action_val = float(action[0]) if isinstance(action, (list, np.ndarray)) else float(action)
        self.last_action = action_val
        self.last_reward = reward

        with open(os.path.join(LOG_DIR, "live_rewards.csv"), "a") as archive_file, \
             open("./csv_log/live_rewards_temp.csv", "a") as tmp_file:
            self.global_step += 1
            archive_file.write(
                f"{self.global_step},{reward},{action_val},{self.timestamp},"
                f"{obs[-4]},{obs[-3]},{obs[-2]},{obs[-1]}\n"
            )
            tmp_file.write(
                f"{self.global_step},{reward},{action_val},{self.timestamp},"
                f"{obs[-4]},{obs[-3]},{obs[-2]},{obs[-1]}\n"
            )

        self.step_count += 1
        terminated = self.step_count >= PARAMS["episode_length"]
        return obs, reward, terminated, False, {}

    # ---------- Robust packet parsing & history-aware receive ----------
    def _parse_packet(self, data_bytes):
        """
        Parse a single UDP packet into a timestamp and a raw observation chunk.

        Parameters
        ----------
        data_bytes : bytes
            Raw UDP payload.

        Returns
        -------
        ts : int
            Parsed timestamp (previous timestamp if parsing fails).
        chunk : np.ndarray
            Raw observation chunk with shape (chunk_size,).
        ok : bool
            True if parsing was successful, False otherwise.
        """
        try:
            parts = data_bytes.decode().strip().split(";")
            if len(parts) < 1 + self.chunk_size:
                return self.timestamp, self._hist[-1].copy(), False
            ts = int(float(parts[0]))
            vals = [float(x) for x in parts[1:1 + self.chunk_size]]
            chunk = np.array(vals, dtype=np.float32)
            return ts, chunk, True
        except Exception:
            return self.timestamp, self._hist[-1].copy(), False

    def _receive_observation(self):
        """
        Receive observation(s) over UDP, optionally discarding initial
        packets (SKIP_FIRST_UDP) and constructing a history-augmented
        observation by concatenating the history buffer.
        """
        # Drain backlog
        sock_recv.setblocking(False)
        while True:
            try:
                sock_recv.recvfrom(100000)
            except BlockingIOError:
                break
            except Exception:
                break
        sock_recv.setblocking(True)

        latest_ts = self.timestamp
        current_chunk = self._hist[-1].copy()

        # Optionally skip first few UDP packets
        for _ in range(max(0, SKIP_FIRST_UDP)):
            data, _ = sock_recv.recvfrom(1000000)
            ts, _, ok = self._parse_packet(data)
            if ok:
                latest_ts = ts

        try:
            data, _ = sock_recv.recvfrom(1000000)
            ts, chunk, ok = self._parse_packet(data)
            if ok:
                latest_ts = ts
                current_chunk = chunk
        except Exception:
            pass

        self._hist.append(current_chunk)
        obs = np.concatenate(list(self._hist), dtype=np.float32)
        self.timestamp = latest_ts
        self.last_obs = obs
        return obs

    # ---------- Rewards ----------
    def _compute_reward_peak_fft_v1(self, obs_pre_reward):
        """
        Compute CTA_1 reward variant from the observation (legacy).
        NOTE: Uses obs[-2] and a global scalar; may require SCALAR_REW definition.
        """
        aux = obs_pre_reward[-2]
        return 1 - aux / SCALAR_REW  # SCALAR_REW must exist if CTA_1 is used

    def _compute_reward_peak_fft_v2(self, obs_pre_reward, Re, alpha, beta):
        """
        Compute CTA_2 reward: combined contribution of FFT peak and meanU
        using a Re-dependent reference.
        """
        aux_fftpeak = obs_pre_reward[-2]
        aux_meanU   = obs_pre_reward[-4]
        mean_term = (0.0002 * Re + 1.4433) - aux_meanU
        return alpha * (1 - aux_fftpeak / SCALAR_REW_FFT) + beta * (mean_term / SCALAR_REW_MEANU)

    def _compute_reward_peak_fft_v3(self, obs_pre_reward, Re, alpha, beta):
        """
        Compute CTA_3 reward: combined contribution of FFT peak and meanU
        with a fixed mean reference (1.5).
        """
        aux_fftpeak = obs_pre_reward[-2]
        aux_meanU   = obs_pre_reward[-4]
        mean_term = 1.5 - aux_meanU
        return alpha * (1 - aux_fftpeak / SCALAR_REW_FFT) + beta * (mean_term / SCALAR_REW_MEANU)

    def _compute_reward_debug_internalUDP(self, obs_pre_reward):
        """
        Debug reward: directly returns the last component of the observation.
        """
        return float(obs_pre_reward[-1])

    def render(self, mode="human"):
        """No-op render (not used)."""
        pass

    def close(self):
        """No-op close (not used)."""
        pass


# ----------------- Env setup (online modes) -----------------
def make_env():
    """
    Factory function to create a Monitor-wrapped CRIOUDPEnv instance.

    Returns
    -------
    gym.Env
        A Monitor-wrapped CRIOUDPEnv.
    """
    env = CRIOUDPEnv()
    return Monitor(env, filename=os.path.join(LOG_DIR, "env_monitor"))


env = DummyVecEnv([make_env])


# ----------------- PPO builder (online training) -----------------
def build_ppo(env_):
    """
    Construct a PPO agent with configuration from PARAMS and attach
    it to the provided vectorized environment.

    Parameters
    ----------
    env_ : VecEnv
        Vectorized environment to train on.

    Returns
    -------
    PPO
        Configured PPO agent (Stable-Baselines3).
    """
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
    run_offloading(mode_train=OFFLOAD_TRAIN)

elif ONLINE_INFER:
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
    # drop-in replacement: EnhancedTensorboardLoggingCallback
    tb_callback = EnhancedTensorboardLoggingCallback(rolling=PARAMS.get("tb_rolling", 500))
    callback = CallbackList([tb_callback, eval_callback])

    if ALGO_TYPE != "PPO":
        raise NotImplementedError("Only PPO supported in this script version.")

    model = build_ppo(env)

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

