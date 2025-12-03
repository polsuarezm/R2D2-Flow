# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/rpo/#rpo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
import socket

# import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

### HYPERPARAMETERS

OBERVATION_LENGTH = 4    # envs.single_observation_space.shape
ACTION_LENGTH = 1    # envs.single_action_space.shape
BATCH_SIZE = 50 + 1   # debe ser al menos (num_steps + 1) ; ademmás en el experimento se pueden descartar los valores iniciales
LAYER1_DIM_ACTOR = 8
LAYER2_DIM_ACTOR = 8
LAYER1_DIM_CRITIC = 16
LAYER2_DIM_CRITIC = 8
LEARNING_RATE = 0.003
GAE_LAMBDA = 0.95
RPO_ALPHA = 0.0 #0.0 significa que RPO se reduce a PPO (usually =0.5)
NUM_MINIBATCHES = 1
GAMMA = 0.99
N_STEPS = 50
N_EPOCH = 5




@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    learning_rate: float = LEARNING_RATE #3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = N_STEPS # 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = GAMMA #0.99
    """the discount factor gamma"""
    gae_lambda: float = GAE_LAMBDA #0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = NUM_MINIBATCHES
    """the number of mini-batches"""
    update_epochs: int = N_EPOCH
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    rpo_alpha: float = RPO_ALPHA
    """the alpha parameter for RPO"""
    # if rpo_alpha = 0.0, RPO reduces to PPO

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def parse_batch(data,device=None):
    N= BATCH_SIZE
    obs_dim= OBERVATION_LENGTH
    action_dim= ACTION_LENGTH
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Validación cabeceras
    if not (data.startswith(b"BATCH") and data.endswith(b"END")):
        raise ValueError("Paquete corrupto o incompleto")

    # 2. Extraer contenido entre BATCH y END
    body = data[len(b"BATCH"):-len(b"END")]
    if body.startswith(b","):
        body = body[1:]

    # 3. Convertir a floats (rápido)
    nums = np.fromstring(body.decode(), sep=",", dtype=np.float32)

    # 4. Cálculo de stride real (según tu parser original)
    stride = obs_dim + 3*action_dim + 1   # 27 + 8 + 1 = 36
    expected = N * stride

    if len(nums) != expected:
        raise ValueError(f"Longitud incorrecta: {len(nums)}, esperado {expected}")

    # 5. Reshape
    nums = nums.reshape(N, stride)

    # 6. Extraer secciones
    idx = 0
    obs_np = nums[:, idx:idx+obs_dim]
    idx += obs_dim

    actions_np = nums[:, idx:idx+action_dim]
    idx += action_dim

    actions_noisy_np = nums[:, idx:idx+action_dim]
    idx += action_dim

    logprobs_np = nums[:, idx:idx+action_dim]
    idx += action_dim

    reward_np = nums[:, idx]  # (N,)
    #idx += 1

    # 7. Convertir TODO a tensores en GPU/CPU apropiada
    b_obs_RT           = torch.tensor(obs_np,          dtype=torch.float32, device=device)
    b_actions_RT       = torch.tensor(actions_np,      dtype=torch.float32, device=device)
    b_actions_noisy_RT = torch.tensor(actions_noisy_np, dtype=torch.float32, device=device)
    b_logprobs_RT = torch.tensor(logprobs_np, dtype=torch.float32, device=device)
    b_reward_RT = torch.tensor(reward_np, dtype=torch.float32, device=device)

    return b_obs_RT, b_actions_RT, b_actions_noisy_RT, b_logprobs_RT, b_reward_RT 


class Agent(nn.Module):
    def __init__(self, rpo_alpha):  #, envs
        super().__init__()
        self.rpo_alpha = rpo_alpha
        self.critic = nn.Sequential(
            layer_init(nn.Linear(OBERVATION_LENGTH, LAYER1_DIM_CRITIC)),
            nn.ReLU(),
            layer_init(nn.Linear(LAYER1_DIM_CRITIC, LAYER2_DIM_CRITIC)),
            nn.ReLU(),
            layer_init(nn.Linear(LAYER2_DIM_CRITIC, 1), std=1.0),  #OUTPUTS REWARD
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(OBERVATION_LENGTH, LAYER1_DIM_ACTOR)),
            nn.ReLU(),
            layer_init(nn.Linear(LAYER1_DIM_ACTOR, LAYER2_DIM_ACTOR)),
            nn.ReLU(),
            layer_init(nn.Linear(LAYER2_DIM_ACTOR, ACTION_LENGTH), std=1.0),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, ACTION_LENGTH)) 

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        else:  # new to RPO
            # sample again to add stochasticity to the policy
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(device)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)
        action_nonoise=action_mean
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x), action_nonoise


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps  # CGG: aumentar numero de steps # int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    #envs = gym.vector.SyncVectorEnv(
    #    [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    #)
    #assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(args.rpo_alpha).to(device)   # envs,
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + (OBERVATION_LENGTH,)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + (ACTION_LENGTH,)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    #next_obs, _ = envs.reset(seed=args.seed)
    #next_obs = torch.Tensor(next_obs).to(device)
    #next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        
        # HERE EXPERIMENT IS LINKED WITH THE PYTHON SCRIPT - RT TARGET INTEGRATION
        
        SEND_TO_PORT = 61558     # puerto donde LabVIEW escucha TCP
        LOCAL_HOST   = "172.17.11.2"

        send_new_deployment = False
        waiting_for_batch_data = True
        batch_data_received = False
        temp_data = b''



        while waiting_for_batch_data:
            try:
                # Crear socket TCP en cada iteración para reconectar si es necesario
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024*1024)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 5*1024*1024)
                sock.settimeout(0.1)  # timeout corto para recv
                sock.connect((LOCAL_HOST, SEND_TO_PORT))
                print("Conectado a LabVIEW")

                while waiting_for_batch_data:
                    try:
                        # Enviar parámetros
                        if send_new_deployment:
                            print("[DEPLOYING NN TO RT TARGET]")
                            params_list = list(agent.actor_mean.parameters())
                            logstd_list = list(agent.actor_logstd.data)
                            params_str = [p.detach().cpu().numpy().tolist() for p in params_list]
                            logstd_str = [p.detach().cpu().numpy().tolist() for p in logstd_list]
                            params_bytes = str(params_str).encode('utf-8')
                            logstd_bytes = str(logstd_str).encode('utf-8')
                            version = str(random.random())
                            sock.sendall(params_bytes + b"LOGSTD" + logstd_bytes + b"VERSION" + version.encode('utf-8') + b"END"+ b"\r\n")  # TCP envía todos los bytes
                            print("[SENT VERSION:", version ,"]")
                            send_new_deployment = False
                        # Recibir respuesta de LabVIEW (si la hay)

                        if batch_data_received:
                            print("[Acknowledging data received]")
                            sock.sendall(b"DATA RECEIVED\r\n")  # TCP envía todos los bytes
                            send_new_deployment = False
                            waiting_for_batch_data = False


                        # Recibir respuesta de LabVIEW (si la hay). OJO: Labviewpuede bloquear el sistema si manda muchos mensajes

                        try:
                            data = sock.recv(5*1024*1024)
                            if not data:
                                break  # LabVIEW cerró la conexión
                            print("[RECV]", data[0:5],".......",data[-30:])
                            match data:
                                case  b'RT waiting for deployment':  #RT target is ready for NN deployment
                                    send_new_deployment = True
                                    print(data)
                                case  b'RT testing NN':  #RT target is testing the last deployment
                                    print("RT testing NN") # here we can add a timeout if needed
                                case  _ :  #Data batch ready
                                    if data[0:5]==b'Deplo':  #batch data ready and received 
                                        send_new_deployment = False
                                        print(data)
                                    if data[0:5]==b'BATCH' and data[-3:]==b'END':  #batch data ready and received 
                                        batch_data_received = True
                                        print("BATCH DATA RECEIVED")
                                        b_obs_RT, b_actions_RT, b_actions_noisy_RT, b_logprobs_RT, b_reward_RT = parse_batch(data,device)
                                    if data[0:5]==b'BATCH' and data[-3:]!=b'END':  #batch data ready and received 
                                        batch_data_received = False
                                        print("HALF BATCH DATA RECEIVED")
                                        temp_data = data;  
                                    if data[0:5]!=b'BATCH' and data[-3:]==b'END':  #batch data ready and received 
                                        batch_data_received = True
                                        print("BATCH DATA RECONSTRUCTED")
                                        data = temp_data + data
                                        b_obs_RT, b_actions_RT, b_actions_noisy_RT, b_logprobs_RT, b_reward_RT = parse_batch(data,device)
                        except socket.timeout:
                            pass  # no hay datos disponibles en este momento

                        time.sleep(0.1)

                    except ConnectionResetError as e:
                        print("Advertencia: LabVIEW cerró la conexión:", e)
                        break  # salir del loop interno y reconectar

                    except Exception as e:
                        print("Error inesperado:", e)
                        time.sleep(0.5)

            except Exception as e:
                print("No se pudo conectar a LabVIEW:", e)
                time.sleep(1)  # esperar antes de reintentar

        print("Let's DRL!!!!")

        for step in range(0, args.num_steps):  #PODEMOS EMPLEAR EL BUCLE QUE HAY PARA COMRPOBAR QUE LA RED SE A APLICADO BIEN y CALCULAR b_values -comprobar tiempo de ejecución.
            global_step += 1 * args.num_envs
            
            next_obs = b_obs_RT[step].unsqueeze(0) #INYECTAMOS VALOR EXPERIMENTAL OBSERVACION . ojo con filas y columnas...
            
            obs[step] = next_obs
            dones[step] = 0. #next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value,action_nonoise = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            
            diff_RT_python = action_nonoise.squeeze(0) - b_actions_RT[step]
            max_diff = torch.max(torch.abs(diff_RT_python)).item()
            if max_diff > 1e-5:
                print("Excess difference between LabVIEW and Python. Check NN:", max_diff,diff_RT_python)       
            
            action = b_actions_noisy_RT[step].unsqueeze(0) #INYECTAMOS VALOR EXPERIMENTAL ACCION (con ruido) . ojo con filas y columnas...
            actions[step] = action
                       
            
            logprobs[step] = b_logprobs_RT[step].unsqueeze(0)
                            # logprob     #este valor debe ser idéntico entre la NN en poython y LabVIEW. nota: no lo estamos comprobando
                                         # XXXXXXXXXXXXXXXX:  Y NO LO ES!!! :XXXXXXXXXXXXXXXXXXXXXXXXX ... está cambiando ¿por qué?
            # TRY NOT TO MODIFY: execute the game and log data.
            #next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            
            reward = b_reward_RT[step+1] # CGG: aquí podemos jugar realmente con el retraso del sistema: el reward es el estado del sistema tras el delay intrínseco
            next_obs = b_obs_RT[step+1].unsqueeze(0) #y añadimos el estado del sistema en la siguiente iteración
            done = [False]      #CGG: nuestro problema nunca termina   # done = np.logical_or(terminations, truncations)
            next_done = torch.Tensor(done).to(device)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            # next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            #if "final_info" in infos:
            #    for info in infos["final_info"]:
            #        if info and "episode" in info:
            #            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            #            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            #            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        
        # bootstrap value if not done  :  ESTA PARTE NOS PERMITE CALCULAR b_advantages b_returns
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + (OBERVATION_LENGTH,))
        b_logprobs = logprobs.reshape((-1,) + (ACTION_LENGTH,))
        b_actions = actions.reshape((-1,) + (ACTION_LENGTH,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # END OF EXPERIMENT INTEGRATION -- RT TARGET INTEGRATION

        # OPTIMIZATION the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                print("EPOCH:",epoch,"END:",end,"START:",start,"IND:",b_inds[start:end])
                _, newlogprob, entropy, newvalue,_ = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
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

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    #envs.close()
    writer.close()
