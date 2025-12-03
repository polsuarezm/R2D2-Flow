"""
RPO / PPO training loop with RT (LabVIEW / cRIO) integration over TCP.

- Receives batched trajectories from a real-time target (RT) via TCP.
- Optionally deploys updated actor parameters back to the RT target.
- Optimizes an RPO/PPO agent on the received batch.
- Supports saving/loading checkpoints for later deployment.
"""

import os
import random
import time
from dataclasses import dataclass
import socket
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


# ==========================
# HYPERPARAMETERS (STATIC)
# ==========================

OBSERVATION_LENGTH = 4      # Dimension of observation from RT target
ACTION_LENGTH = 1           # Dimension of action sent to RT target
BATCH_SIZE = 50 + 1         # Must be >= (num_steps + 1); extra step for system delay

LAYER1_DIM_ACTOR = 8
LAYER2_DIM_ACTOR = 8
LAYER1_DIM_CRITIC = 16
LAYER2_DIM_CRITIC = 8

LEARNING_RATE = 0.003
GAE_LAMBDA = 0.95
RPO_ALPHA = 0.0  # RPO_ALPHA = 0.0 => RPO reduces to PPO (typical RPO_ALPHA ~ 0.5)
NUM_MINIBATCHES = 1
GAMMA = 0.7
N_STEPS = 50
N_EPOCH = 5


# ==========================
# ARGUMENTS
# ==========================

@dataclass
class Args:
    """
    Command-line arguments for this experiment.
    Many of these come from the original CleanRL RPO implementation; some
    environment-related features are unused in this RT integration version.
    """
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """Name of this experiment (used in run_name)."""

    seed: int = 1
    """Random seed for reproducibility."""

    torch_deterministic: bool = True
    """If True, set CUDNN deterministic mode (slower but reproducible)."""

    cuda: bool = False
    """If True and CUDA is available, use GPU."""

    track: bool = False
    """If True, log to Weights & Biases in addition to TensorBoard."""

    wandb_project_name: str = "cleanRL"
    """Weights & Biases project name."""

    wandb_entity: str | None = None
    """Weights & Biases entity (team or user)."""

    capture_video: bool = False
    """Unused in RT mode; kept for compatibility with CleanRL template."""

    # Algorithm-specific arguments
    env_id: str = "HalfCheetah-v4"
    """Environment ID (unused in RT mode, kept for compatibility)."""

    total_timesteps: int = 8_000_000
    """Total timesteps for the experiment (used to compute number of updates)."""

    learning_rate: float = LEARNING_RATE
    """Optimizer learning rate."""

    num_envs: int = 1
    """Number of parallel environments (fixed to 1 in RT mode)."""

    num_steps: int = N_STEPS
    """Number of steps used per policy rollout / batch (from RT target)."""

    anneal_lr: bool = True
    """Enable linear learning rate annealing."""

    gamma: float = GAMMA
    """Discount factor gamma."""

    gae_lambda: float = GAE_LAMBDA
    """Lambda parameter for Generalized Advantage Estimation (GAE)."""

    num_minibatches: int = NUM_MINIBATCHES
    """Number of mini-batches per update."""

    update_epochs: int = N_EPOCH
    """Number of epochs (K) over each batch."""

    norm_adv: bool = True
    """Normalize advantages if True."""

    clip_coef: float = 0.2
    """PPO surrogate clipping coefficient."""

    clip_vloss: bool = True
    """Use clipped value loss if True."""

    ent_coef: float = 0.0
    """Entropy coefficient."""

    vf_coef: float = 0.5
    """Value function loss coefficient."""

    max_grad_norm: float = 0.5
    """Max gradient norm for clipping."""

    target_kl: float | None = None
    """Early-stopping threshold on approximate KL divergence; disable if None."""

    rpo_alpha: float = RPO_ALPHA
    """RPO alpha parameter; if 0.0, this reduces to PPO behavior."""

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    """Base directory to store model checkpoints."""

    save_interval: int = 50
    """Save a checkpoint every this many updates (0 disables periodic saving)."""

    load_model_path: str | None = None
    """If set, load a model checkpoint from this path before training/deployment."""

    # To be filled in at runtime:
    batch_size: int = 0
    """Flattened batch size: num_envs * num_steps."""

    minibatch_size: int = 0
    """Mini-batch size for SGD."""

    num_iterations: int = 0
    """Number of iterations, derived from total_timesteps and batch_size."""


# ==========================
# UTILITIES
# ==========================

def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """
    Initialize a linear layer with orthogonal weights and constant bias.

    Args:
        layer: Linear layer to be initialized.
        std: Standard deviation for orthogonal initialization.
        bias_const: Constant value for the bias.

    Returns:
        The same layer with initialized weights and bias.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def parse_batch(data: bytes, device: torch.device | None = None):
    """
    Parse a batch of RT data coming from LabVIEW over TCP.

    The packet format is:
        b"BATCH" + CSV_DATA + b"END"
    where CSV_DATA corresponds to N rows of:
        [obs_dim, action_dim, action_dim (noisy), action_dim (logprob), reward]

    Args:
        data: Raw bytes received from the RT target (LabVIEW).
        device: Torch device where tensors will be allocated. If None,
                it will be set to CUDA if available, otherwise CPU.

    Returns:
        Tuple of tensors:
            (obs, actions, actions_noisy, logprobs, rewards)
        with shapes compatible with the training loop.
    """
    N = BATCH_SIZE
    obs_dim = OBSERVATION_LENGTH
    action_dim = ACTION_LENGTH

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Validate headers
    if not (data.startswith(b"BATCH") and data.endswith(b"END")):
        raise ValueError("Corrupted or incomplete packet: missing BATCH/END markers.")

    # 2. Extract the body between 'BATCH' and 'END'
    body = data[len(b"BATCH"):-len(b"END")]
    if body.startswith(b","):
        body = body[1:]

    # 3. Convert to float32 array
    nums = np.fromstring(body.decode(), sep=",", dtype=np.float32)

    # 4. Compute stride and expected length
    stride = obs_dim + 3 * action_dim + 1
    expected = N * stride
    if len(nums) != expected:
        raise ValueError(f"Incorrect payload length: got {len(nums)}, expected {expected}")

    # 5. Reshape: (N, stride)
    nums = nums.reshape(N, stride)

    # 6. Extract sections
    idx = 0
    obs_np = nums[:, idx: idx + obs_dim]
    idx += obs_dim

    actions_np = nums[:, idx: idx + action_dim]
    idx += action_dim

    actions_noisy_np = nums[:, idx: idx + action_dim]
    idx += action_dim

    logprobs_np = nums[:, idx: idx + action_dim]
    idx += action_dim

    reward_np = nums[:, idx]  # shape: (N,)

    # 7. Convert to tensors on the chosen device
    b_obs_RT = torch.tensor(obs_np, dtype=torch.float32, device=device)
    b_actions_RT = torch.tensor(actions_np, dtype=torch.float32, device=device)
    b_actions_noisy_RT = torch.tensor(actions_noisy_np, dtype=torch.float32, device=device)
    b_logprobs_RT = torch.tensor(logprobs_np, dtype=torch.float32, device=device)
    b_reward_RT = torch.tensor(reward_np, dtype=torch.float32, device=device)

    return b_obs_RT, b_actions_RT, b_actions_noisy_RT, b_logprobs_RT, b_reward_RT


# ==========================
# AGENT
# ==========================

class Agent(nn.Module):
    """
    RPO/PPO agent with a shared critic and mean-action actor.

    - Critic: MLP mapping observation -> scalar value.
    - Actor: MLP producing mean action + learnable log-std.
    - RPO-style additional stochasticity is optionally applied to the mean action.
    """

    def __init__(self, rpo_alpha: float):
        """
        Initialize the agent architecture.

        Args:
            rpo_alpha: RPO alpha parameter controlling additional mean noise.
                       If 0.0, the method reduces to standard PPO behavior.
        """
        super().__init__()
        self.rpo_alpha = rpo_alpha

        # Critic network: observation -> scalar value
        self.critic = nn.Sequential(
            layer_init(nn.Linear(OBSERVATION_LENGTH, LAYER1_DIM_CRITIC)),
            nn.ReLU(),
            layer_init(nn.Linear(LAYER1_DIM_CRITIC, LAYER2_DIM_CRITIC)),
            nn.ReLU(),
            layer_init(nn.Linear(LAYER2_DIM_CRITIC, 1), std=1.0),
        )

        # Actor network: observation -> mean action
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(OBSERVATION_LENGTH, LAYER1_DIM_ACTOR)),
            nn.ReLU(),
            layer_init(nn.Linear(LAYER1_DIM_ACTOR, LAYER2_DIM_ACTOR)),
            nn.ReLU(),
            layer_init(nn.Linear(LAYER2_DIM_ACTOR, ACTION_LENGTH), std=1.0),
        )

        # Log-standard deviation of the action distribution (learned)
        self.actor_logstd = nn.Parameter(torch.zeros(1, ACTION_LENGTH) - 1.0)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the state-value estimate V(s).

        Args:
            x: Batch of observations of shape (batch_size, OBSERVATION_LENGTH).

        Returns:
            Tensor of shape (batch_size, 1) with value estimates.
        """
        return self.critic(x)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        """
        Compute action, log-probability, entropy, value, and noiseless action mean.

        If `action` is None:
            - Sample an action from the current policy distribution.

        If `action` is provided (RPO mode):
            - Apply additional uniform noise to the mean (controlled by rpo_alpha),
              construct a new distribution, and re-evaluate log-prob and value
              for the given `action`.

        Args:
            x: Observations, shape (batch_size, OBSERVATION_LENGTH).
            action: Optional actions to be evaluated by the policy.

        Returns:
            Tuple:
                action: Sampled or provided action, shape (batch_size, ACTION_LENGTH).
                log_prob: Log-probability of the action under the current policy.
                entropy: Entropy of the action distribution.
                value: Critic value, V(s).
                action_nonoise: Mean action before RPO noise (for RT consistency check).
        """
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            # Standard PPO mode: sample from the current policy
            action = probs.sample()
        else:
            # RPO mode: perturb the mean with uniform noise in [-alpha, alpha]
            device = action_mean.device
            z = torch.empty_like(action_mean, device=device).uniform_(-self.rpo_alpha, self.rpo_alpha)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)

        action_nonoise = action_mean
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        value = self.critic(x)

        return action, log_prob, entropy, value, action_nonoise


# ==========================
# CHECKPOINT UTILITIES
# ==========================

def save_checkpoint(path: str, agent: Agent, optimizer: optim.Optimizer,
                    args: Args, global_step: int, update: int):
    """
    Save model + optimizer state so we can resume training or redeploy later.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "global_step": global_step,
        "update": update,
    }
    torch.save(checkpoint, path)
    print(f"[CHECKPOINT] Saved to {path}")


def load_checkpoint(path: str, agent: Agent, optimizer: optim.Optimizer, device: torch.device):
    """
    Load model (and optimizer if present) from checkpoint.

    Returns:
        global_step, update (if stored), so you can resume if desired.
    """
    print(f"[CHECKPOINT] Loading from {path}")
    checkpoint = torch.load(path, map_location=device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    global_step = checkpoint.get("global_step", 0)
    update = checkpoint.get("update", 0)
    print(f"[CHECKPOINT] Loaded (global_step={global_step}, update={update})")
    return global_step, update


# ==========================
# MAIN SCRIPT
# ==========================

if __name__ == "__main__":
    args = tyro.cli(Args)

    # Derived sizes
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Where to store checkpoints for this run
    run_checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)

    # Optional Weights & Biases logging
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # --- CSV logging for reward vs step/action ---
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    csv_log_path = f"runs/{run_name}/step_log.csv"
    csv_file = open(csv_log_path, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    # header row
    csv_writer.writerow(
        [
            "global_step",
            "update",
            "step_in_update",
            "reward",
            "action_noisy",
            "action_mean",
            "value",
        ]
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Agent and optimizer
    agent = Agent(args.rpo_alpha).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # If requested, load a pre-trained model (for warm-start or pure deployment)
    global_step = 0
    resume_update = 0
    if args.load_model_path is not None:
        if os.path.isfile(args.load_model_path):
            global_step, resume_update = load_checkpoint(args.load_model_path, agent, optimizer, device)
        else:
            print(f"[CHECKPOINT] WARNING: {args.load_model_path} not found, starting from scratch.")

    # Storage tensors (PPO-style rollout buffers)
    obs = torch.zeros((args.num_steps, args.num_envs, OBSERVATION_LENGTH), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs, ACTION_LENGTH), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    start_time = time.time()
    num_updates = args.total_timesteps // args.batch_size

    # RT / LabVIEW TCP config
    SEND_TO_PORT = 61558  # TCP port where LabVIEW is listening
    LOCAL_HOST = "172.17.11.2"

    for update in range(1, num_updates + 1):
        # Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # ===============================
        # RT TARGET INTEGRATION (TCP)
        # ===============================
        send_new_deployment = False
        waiting_for_batch_data = True
        batch_data_received = False
        temp_data = b""

        # Receive a full BATCH...END payload from LabVIEW
        while waiting_for_batch_data:
            try:
                # Create a fresh TCP socket each (re)connection attempt
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 5 * 1024 * 1024)
                sock.settimeout(0.1)
                sock.connect((LOCAL_HOST, SEND_TO_PORT))
                print("Connected to LabVIEW.")

                while waiting_for_batch_data:
                    try:
                        # If LabVIEW requested a deployment, send NN parameters
                        if send_new_deployment:
                            print("[DEPLOYING NN TO RT TARGET]")
                            params_list = list(agent.actor_mean.parameters())
                            logstd_list = [agent.actor_logstd.data]

                            params_str = [p.detach().cpu().numpy().tolist() for p in params_list]
                            logstd_str = [p.detach().cpu().numpy().tolist() for p in logstd_list]

                            params_bytes = str(params_str).encode("utf-8")
                            logstd_bytes = str(logstd_str).encode("utf-8")
                            version = str(random.random())

                            sock.sendall(
                                params_bytes
                                + b"LOGSTD"
                                + logstd_bytes
                                + b"VERSION"
                                + version.encode("utf-8")
                                + b"END"
                                + b"\r\n"
                            )
                            print("[SENT VERSION:", version, "]")
                            send_new_deployment = False

                        # If we already reconstructed a batch, acknowledge and exit loop
                        if batch_data_received:
                            print("[Acknowledging data received]")
                            sock.sendall(b"DATA RECEIVED\r\n")
                            waiting_for_batch_data = False

                        # Try to receive data from LabVIEW (non-blocking via timeout)
                        try:
                            data = sock.recv(5 * 1024 * 1024)
                            if not data:
                                # LabVIEW closed the connection
                                break

                            print("[RECV]", data[0:5], ".......", data[-30:])
                            match data:
                                case b"RT waiting for deployment":
                                    # RT target is ready to receive a new NN deployment
                                    send_new_deployment = True
                                    print(data)
                                case b"RT testing NN":
                                    # RT target is currently testing the last deployment
                                    print("RT testing NN.")
                                case _:
                                    # Data batch or other messages
                                    if data[0:5] == b"Deplo":
                                        # Information-only deployment messages
                                        print(data)

                                    if data[0:5] == b"BATCH" and data[-3:] == b"END":
                                        # Complete batch in a single packet
                                        batch_data_received = True
                                        print("BATCH DATA RECEIVED")
                                        (
                                            b_obs_RT,
                                            b_actions_RT,
                                            b_actions_noisy_RT,
                                            b_logprobs_RT,
                                            b_reward_RT,
                                        ) = parse_batch(data, device)

                                    elif data[0:5] == b"BATCH" and data[-3:] != b"END":
                                        # First half of a split batch
                                        batch_data_received = False
                                        print("HALF BATCH DATA RECEIVED")
                                        temp_data = data

                                    elif data[0:5] != b"BATCH" and data[-3:] == b"END":
                                        # Second half of a split batch
                                        batch_data_received = True
                                        print("BATCH DATA RECONSTRUCTED")
                                        data = temp_data + data
                                        (
                                            b_obs_RT,
                                            b_actions_RT,
                                            b_actions_noisy_RT,
                                            b_logprobs_RT,
                                            b_reward_RT,
                                        ) = parse_batch(data, device)
                        except socket.timeout:
                            # No data available at this moment
                            pass

                        time.sleep(0.1)

                    except ConnectionResetError as e:
                        print("Warning: LabVIEW closed the connection:", e)
                        break

                    except Exception as e:
                        print("Unexpected error inside TCP loop:", e)
                        time.sleep(0.5)

            except Exception as e:
                print("Could not connect to LabVIEW:", e)
                time.sleep(1.0)

        print("Starting PPO/RPO update from RT batch...")

        # ===================================
        # ROLLOUT CONSTRUCTION FROM RT DATA
        # ===================================
        for step in range(args.num_steps):
            global_step += 1 * args.num_envs

            # Inject experimental observation from RT target
            next_obs = b_obs_RT[step].unsqueeze(0)  # shape: (1, OBSERVATION_LENGTH)
            obs[step] = next_obs
            dones[step] = 0.0  # RT problem never truly "terminates" in this setup

            # Get PPO/RPO action and value for the current observation
            with torch.no_grad():
                action_model, logprob_model, _, value, action_nonoise = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            # Consistency check between RT LabVIEW action and Python NN (noiseless)
            diff_RT_python = action_nonoise.squeeze(0) - b_actions_RT[step]
            max_diff = torch.max(torch.abs(diff_RT_python)).item()
            if max_diff > 1e-5:
                print(
                    "Warning: Difference between LabVIEW and Python NN is larger than tolerance:",
                    max_diff,
                    diff_RT_python,
                )

            # Inject experimental noisy action and logprob from RT target
            action = b_actions_noisy_RT[step].unsqueeze(0)  # shape: (1, ACTION_LENGTH)
            actions[step] = action
            logprobs[step] = b_logprobs_RT[step].unsqueeze(0)

            # Reward is taken from the next state (system delay modeling)
            reward = b_reward_RT[step + 1]          # already a tensor on correct device
            next_obs = b_obs_RT[step + 1].unsqueeze(0)
            done = [False]                          # RT never ends
            next_done = torch.tensor(done, device=device, dtype=torch.float32)

            rewards[step] = reward.view(-1)

            # --- CSV logging: one row per (update, step) ---
            action_noisy_scalar = float(action.squeeze().detach().cpu().numpy())
            action_mean_scalar = float(action_nonoise.squeeze().detach().cpu().numpy())
            value_scalar = float(values[step].detach().cpu().item())
            reward_scalar = float(reward.detach().cpu().item())

            csv_writer.writerow(
                [
                    global_step,
                    update,
                    step,
                    reward_scalar,
                    action_noisy_scalar,
                    action_mean_scalar,
                    value_scalar,
                ]
            )

        # ===================
        # GAE / RETURNS
        # ===================
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0.0

            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam

            returns = advantages + values

        # Flatten batch: (num_steps * num_envs, ...)
        b_obs = obs.reshape((-1, OBSERVATION_LENGTH))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, ACTION_LENGTH))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # ===================
        # PPO / RPO UPDATE
        # ===================
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # ===================
        # METRICS & LOGGING
        # ===================
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Basic scalars
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        # Extra diagnostics from this RT batch
        writer.add_scalar("batch/reward_mean", rewards.mean().item(), global_step)
        writer.add_scalar("batch/reward_std", rewards.std().item(), global_step)
        writer.add_scalar("batch/value_mean", values.mean().item(), global_step)
        writer.add_scalar("batch/value_std", values.std().item(), global_step)
        writer.add_scalar("batch/adv_mean", b_advantages.mean().item(), global_step)
        writer.add_scalar("batch/adv_std", b_advantages.std().item(), global_step)
        writer.add_scalar("batch/action_mean", b_actions.mean().item(), global_step)
        writer.add_scalar("batch/action_std", b_actions.std().item(), global_step)
        writer.add_scalar("batch/logprob_mean", b_logprobs.mean().item(), global_step)
        writer.add_scalar("batch/logprob_std", b_logprobs.std().item(), global_step)
        writer.add_scalar(
            "batch/actor_logstd_mean", agent.actor_logstd.detach().mean().item(), global_step
        )

        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)
        writer.add_scalar("charts/SPS", sps, global_step)

        # -------------------
        # Save checkpoint
        # -------------------
        if args.save_interval > 0 and (update % args.save_interval == 0):
            ckpt_path = os.path.join(run_checkpoint_dir, f"model_update_{update:06d}.pt")
            save_checkpoint(ckpt_path, agent, optimizer, args, global_step, update)

    csv_file.close()
    writer.close()
