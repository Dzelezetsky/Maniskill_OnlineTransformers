
from collections import defaultdict
from dataclasses import dataclass
import os
import random
import time
from typing import Optional
from torch.utils.data import TensorDataset, DataLoader
import tqdm
import math

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tyro

import mani_skill.envs


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "SAC"
    """the group of the run for wandb"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_trajectory: bool = False
    """whether to save trajectory data into the `videos` folder"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    log_freq: int = 1_000
    """logging frequency in terms of environment steps"""

    # Environment specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    env_vectorization: str = "gpu"
    """the type of environment vectorization to use"""
    num_envs: int = 32
    """the number of parallel environments"""
    num_eval_envs: int = 32
    """the number of parallel evaluation environments"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""

    # Algorithm specific arguments
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    buffer_size: int = 1_000_000
    """the replay memory buffer size"""
    buffer_device: str = "cuda"
    """where the replay buffer is stored. Can be 'cpu' or 'cuda' for GPU"""
    gamma: float = 0.8
    """the discount factor gamma"""
    tau: float = 0.01
    """target smoothing coefficient"""
    batch_size: int = 1024
    """the batch size of sample from the replay memory"""
    learning_starts: int = 4_000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    training_freq: int = 64
    """training frequency (in steps)"""
    utd: float = 0.5
    """update to data ratio"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    bootstrap_at_done: str = "always"
    """the bootstrap method to use when a done signal is received. Can be 'always' or 'never'"""

    # to be filled in runtime
    grad_steps_per_iteration: int = 0
    """the number of gradient updates per iteration"""
    steps_per_env: int = 0
    """the number of steps each parallel env takes per iteration"""
    
    ##### SPECIAL ARGS FOR TRANSFORMER  ##########
    
    use_gates: bool = False
    """use gatings instead skip connection in transformer block"""
    n_embd: int = 228  #483#256  
    """inner transformer dimention"""
    n_layer: int = 1
    n_head: int = 2
    dropout: float = 0.0
    seq_len: int = 5
    bias: bool = True
    loss_proportion: str = 'bc'
    use_train_data: bool = True
    

@dataclass
class ReplayBufferSample:
    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor

class ReplayBuffer:
    def __init__(self, env, num_envs: int, buffer_size: int,
                 storage_device: torch.device, sample_device: torch.device):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device
        self.per_env_buffer_size = buffer_size // num_envs
        print(self.per_env_buffer_size)

        # основной буфер
        self.obs          = torch.zeros((self.per_env_buffer_size, num_envs) + env.single_observation_space.shape, device=storage_device)
        self.next_obs     = torch.zeros((self.per_env_buffer_size, num_envs) + env.single_observation_space.shape, device=storage_device)
        self.actions      = torch.zeros((self.per_env_buffer_size, num_envs) + env.single_action_space.shape,   device=storage_device)
        self.rewards      = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        self.dones        = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)

        # доп. статистики
        self.associated_r = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        self.Q            = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        self.V            = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)

    
    def add(self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor):
        if self.storage_device == torch.device("cpu"):
            obs = obs.cpu()
            next_obs = next_obs.cpu()
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()
        
        if self.pos+args.num_steps >= self.per_env_buffer_size+2:
            self.pos = 0
            
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs

        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0
            
        return self.pos-1    
    
    def sample(self, batch_size: int):
        if self.full:
            batch_inds = torch.randint(0, self.per_env_buffer_size, size=(batch_size, ))
        else:
            batch_inds = torch.randint(0, self.pos, size=(batch_size, ))
        env_inds = torch.randint(0, self.num_envs, size=(batch_size, ))
        return ReplayBufferSample(
            obs=self.obs[batch_inds, env_inds].to(self.sample_device),
            next_obs=self.next_obs[batch_inds, env_inds].to(self.sample_device),
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device)
        )
    
    
    def add_associated_reward(self, positions: list[int], rewards: torch.Tensor):
        """
        Добавляет associated_r, корректно обрабатывая wrap-around.
        """
        if not positions:
            return

        # нормализуем позиции по модулю размера буфера
        buf = self.per_env_buffer_size
        pos = torch.tensor([p % buf for p in positions], device=self.storage_device, dtype=torch.long)  # [T]

        # расширяем rewards: [1, N] → [T, N]
        R = rewards.to(self.storage_device).unsqueeze(0).expand(len(pos), -1)

        # аккуратно добавляем
        self.associated_r[pos, :] += R


    def add_qv_estimates(self, positions: list[int]):
        """
        Вычисляет Q/V, учитывая wrap-around через модуль и хронологический порядок.
        """
        if not positions:
            return

        # 1) нормализуем и ищем wrap-point
        buf = self.per_env_buffer_size
        norm = [p % buf for p in positions]
        split = next((i+1 for i in range(len(norm)-1) if norm[i] > norm[i+1]), None)

        # 2) упорядочиваем в хронологическом порядке
        ordered = (norm[split:] + norm[:split]) if split is not None else norm[:]

        # 3) собираем индексы
        pos = torch.tensor(ordered, device=self.storage_device, dtype=torch.long)  # [T]
        env = torch.arange(self.num_envs, device=self.storage_device)              # [N]
        ti, ei = torch.meshgrid(pos, env, indexing="ij")                           # [T, N]

        # 4) берём R и A
        R = self.rewards[ti, ei]         # [T, N]
        A = self.associated_r[ti, ei]    # [T, N]

        # 5) считаем Q/V
        cum_excl = torch.cumsum(R, dim=0) - R
        cum_incl = torch.cumsum(R, dim=0)

        Qv = A - cum_excl
        Vv = A - cum_incl

        # 6) записываем обратно
        self.Q[pos, :] = Qv
        self.V[pos, :] = Vv


    def make_sequential_dataloader(self,
                                   positions: list[int],
                                   context_len: int,
                                   batch_size: int,
                                   shuffle: bool = True):
        """
        Собирает скользящие окна, учитывая wrap-around.
        """
        if len(positions) < context_len:
            raise ValueError("len(positions) < context_len")

        # 1) нормализуем и находим wrap-point
        buf = self.per_env_buffer_size
        norm = [p % buf for p in positions]
        split = next((i+1 for i in range(len(norm)-1) if norm[i] > norm[i+1]), None)

        # 2) упорядочиваем
        ordered = (norm[split:] + norm[:split]) if split is not None else norm[:]

        # 3) собираем весь временной ряд [T,N,...]
        pos = torch.tensor(ordered, device=self.storage_device, dtype=torch.long)
        env = torch.arange(self.num_envs, device=self.storage_device)
        ti, ei = torch.meshgrid(pos, env, indexing="ij")

        obs_seq      = self.obs[ti, ei]
        next_obs_seq = self.next_obs[ti, ei]
        act_seq      = self.actions[ti, ei]
        rew_seq      = self.rewards[ti, ei]
        done_seq     = self.dones[ti, ei]
        ar_seq       = self.associated_r[ti, ei]
        Q_seq        = self.Q[ti, ei]
        V_seq        = self.V[ti, ei]

        # 4) строим скользящие окна и раскладываем по env
        sequences = []
        T = pos.size(0)
        for start in range(T - context_len + 1):
            end = start + context_len

            o = obs_seq[start:end].permute(1, 0, *range(2, obs_seq.ndim)).to(self.sample_device)
            no = next_obs_seq[start:end].permute(1, 0, *range(2, next_obs_seq.ndim)).to(self.sample_device)
            a = act_seq[start:end].permute(1, 0, *range(2, act_seq.ndim)).to(self.sample_device)
            r = rew_seq[start:end].permute(1, 0).to(self.sample_device)
            d = done_seq[start:end].permute(1, 0).to(self.sample_device)
            ar = ar_seq[start:end].permute(1, 0).to(self.sample_device)
            Qv = Q_seq[start:end].permute(1, 0).to(self.sample_device)
            Vv = V_seq[start:end].permute(1, 0).to(self.sample_device)

            for e in range(self.num_envs):
                sequences.append((o[e], no[e], a[e], r[e], d[e], ar[e], Qv[e], Vv[e]))

        # 5) упаковываем в DataLoader
        o_b, no_b, a_b, r_b, d_b, ar_b, Q_b, V_b = zip(*sequences)
        dataset = TensorDataset(
            torch.stack(o_b),
            torch.stack(no_b),
            torch.stack(a_b),
            torch.stack(r_b),
            torch.stack(d_b),
            torch.stack(ar_b),
            torch.stack(Q_b),
            torch.stack(V_b),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)   
    
    
########## SPECIAL TOOLS FOR TRANSFORMER BELOW  ##################

class PositionalEncoding(nn.Module):
    def __init__(self, dim, device, min_timescale=2.0, max_timescale=1e4):
        super().__init__()
        self.device = device
        freqs = torch.arange(0, dim, min_timescale).to(self.device)
        inv_freqs = max_timescale ** (-freqs / dim)
        self.register_buffer("inv_freqs", inv_freqs)

    def forward(self, seq_len):
        seq = torch.arange(seq_len - 1, -1, -1.0).to(self.device)
        sinusoidal_inp = rearrange(seq, "n -> n ()") * rearrange(self.inv_freqs, "d -> () d")
        pos_emb = torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim=-1)
        return pos_emb



class GRUGate(nn.Module):

    def __init__(self, input_dim: int, bg: float = 0.0):
        
        super(GRUGate, self).__init__()
        self.Wr = nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = nn.Linear(input_dim, input_dim, bias=False)
        self.bg = nn.Parameter(torch.full([input_dim], bg))  # bias
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        nn.init.xavier_uniform_(self.Wr.weight)
        nn.init.xavier_uniform_(self.Ur.weight)
        nn.init.xavier_uniform_(self.Wz.weight)
        nn.init.xavier_uniform_(self.Uz.weight)
        nn.init.xavier_uniform_(self.Wg.weight)
        nn.init.xavier_uniform_(self.Ug.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """        
        Arguments:
            x {torch.tensor} -- First input
            y {torch.tensor} -- Second input
        Returns:
            {torch.tensor} -- Output
        """
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))

        # print(f'mean z: {z.mean()}')

        return torch.mul(1 - z, x) + torch.mul(z, h) #, z.mean()



class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.num_steps, config.num_steps))
                                        .view(1, 1, config.num_steps, config.num_steps))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y



class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

        if config.use_gates:
            self.skip_fn_1 = GRUGate(config.n_embd, 2.0)
            self.skip_fn_2 = GRUGate(config.n_embd, 2.0)
        else:
            self.skip_fn_1 = lambda x, y: x + y
            self.skip_fn_2 = lambda x, y: x + y


    def forward(self, x):

        x = self.skip_fn_1(x, self.attn(self.ln_1(x)))
        x = self.skip_fn_2(x, self.mlp(self.ln_2(x)))

        
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.pos_embedding = nn.Embedding(config.max_episode_steps, config.n_embd)

        self.transformer_layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

        
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    

    def get_num_params(self, non_embedding=True):
        
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):

        t = x.shape[1]

        device = x.device
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        pos_emb = self.pos_embedding(pos) # position embeddings of shape (t, n_embd)
        
        x = self.drop(x + pos_emb)
        for block in self.transformer_layers:
            x = block(x)
        x = self.ln_f(x)

        return x
    
class Trans_Actor(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.transformer = GPT(args)
        self.encoder = nn.Linear(np.array(env.single_observation_space.shape).prod(), args.n_embd)
        
        self.head = nn.Sequential(
            nn.Linear(args.n_embd, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))
        # will be saved in the state_dict

    def forward(self, x):
        x = self.encoder(x)
        x = self.transformer(x)[:,-1,:]
        x = self.head(x)
        
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std   ### !!! squeeze!

    def get_eval_action(self, x):
        x = self.encoder(x)
        x = self.transformer(x)[:, -1, :]
        x = self.head(x)
        
        mean = self.fc_mean(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action   ### !!! squeeze!

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)
# ALGO LOGIC: initialize agent here:
class Trans_SoftQNetwork(nn.Module):
    '''
    Q-network for Transformer-based maniskill tasks
    '''
    def __init__(self, env, args):
        super().__init__()
        
        self.transformer = GPT(args)
        self.encoder = nn.Linear(np.array(env.single_observation_space.shape).prod(), args.n_embd)
        self.net = nn.Sequential(
            nn.Linear(args.n_embd + np.prod(env.single_action_space.shape), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, a):                # x = (batch,cont,s_d)   a = (batch,a_d)
        x = self.encoder(x)                 # x = (batch,cont,n_embd)
        x = self.transformer(x)[:, -1, :]   # x = (batch,n_embd)
        x = torch.cat([x, a], 1)            # x = (batch,n_embd+a_d)
        return self.net(x)                  # x = (batch,1)



########## SPECIAL TOOLS FOR TRANSFORMER ABOVE  ##################  

# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.net(x)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))
        # will be saved in the state_dict

    def forward(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_eval_action(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)

class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()
        

def train_transformer(loss_proportion: str, positions: list, ascent_on: str, batch_size: int, context: int, shuffle: bool):
    
    '''
    obs       torch.Size([bs, cont, s_d]) 
    n_obs     torch.Size([bs, cont, s_d]) 
    acts      torch.Size([bs, cont, 4]) 
    rew       torch.Size([bs, cont]) 
    dones     torch.Size([bs, cont]) 
    R         torch.Size([bs, cont]) 
    Q         torch.Size([bs, cont]) 
    V         torch.Size([bs, cont])
    '''
    start_time = time.time()
    dataloader = rb.make_sequential_dataloader(positions=positions, context_len=context, batch_size=batch_size, shuffle=shuffle)
    #torch.autograd.set_detect_anomaly(True)
    for batch in dataloader:
        obs = batch[0] # state_based obs
        n_obs = batch[1] # state_based next obs
        acts = batch[2]
        rew = batch[3]
        dones = batch[4]
        R = batch[5]  # Associated rewards with the episode under the positions
        Q = batch[6]  # Real Q values associated with states and observation on the positions
        V = batch[7]  # Real V values

        
        if loss_proportion in ['rl+bc', 'bc']:        
        
            #ACTOR BC LOSS
            predicted_action, _, _= trans_actor.get_action(obs)
            target_action = acts[:, -1].clone()  
            criterion = nn.MSELoss()
            trans_actor_BC_loss = criterion(predicted_action, target_action)
            
            trans_actor_optimizer.zero_grad()
            trans_actor_BC_loss.backward()
            trans_actor_optimizer.step()
            
            
            # CRITIC BC LOSS
            action_target = acts[:, -1].clone()

            q_predicted1 = trans_qf1(obs, action_target)
            q_predicted2 = trans_qf2(obs, action_target)

            # Only last timestep
            with torch.no_grad():
                q_target1 = qf1(obs[:,-1,], action_target)
                q_target2 = qf2(obs[:,-1,], action_target)

            trans_qf1_loss = F.mse_loss(q_predicted1, q_target1)
            trans_qf2_loss = F.mse_loss(q_predicted2, q_target2)
            trans_critic_BC_loss = trans_qf1_loss + trans_qf2_loss
            
            trans_q_optimizer.zero_grad()
            trans_critic_BC_loss.backward()
            trans_q_optimizer.step()

            logger.add_scalar("Trans/Actor_BC_loss", trans_actor_BC_loss.item(), global_step)
            logger.add_scalar("Trans/Critic_BC_loss", trans_critic_BC_loss.item(), global_step)
        
        if loss_proportion in ['rl+bc', 'rl']: 
        
            #ACTOR RL LOSS
            pi, log_pi, _ = trans_actor.get_action(obs)
            
            pi_detached = pi.clone() #?????????????
            if ascent_on == 'transformer':
                qf1_pi = trans_qf1(obs , pi_detached) #detach_encoder=True 
                qf2_pi = trans_qf2(obs , pi_detached) #detach_encoder=True 
            elif ascent_on == 'accelerator':
                qf1_pi = qf1(obs[:,-1,], pi_detached) #detach_encoder=True 
                qf2_pi = qf2(obs[:,-1,], pi_detached) #detach_encoder=True 
            
            min_qf_pi = torch.min(qf1_pi, qf2_pi)#.view(-1)
            
            trans_actor_RL_loss = ((alpha * log_pi) - min_qf_pi).mean()
            
            trans_actor_optimizer.zero_grad()
            trans_actor_RL_loss.backward()
            trans_actor_optimizer.step()
            
            #CRITIC RL LOSS
            '''
            пересчёт R позволяет в том числе ещё и убрать второго критика (на момент разгона)
            но потом при файнтюне он снова понадобится так как у нас больше не будет R
            таргеты можно сразу убрать и на момент файнтюна инициализировать копиями критиков
            '''
            #Q_target = Q[:, -1].clone()
            action_target = acts[:, -1].clone()
            
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action( n_obs[:,-1,] )  # a* = \pi(s') 
                #print(f"Из n_obs {n_obs.shape} получили next_state_actions {next_state_actions.shape}")
                qf1_next_target = trans_qf1_target(n_obs, next_state_actions)
                qf2_next_target = trans_qf2_target(n_obs, next_state_actions)   # Q(s',a*)
                #print(f"Транс таргет критики выдали  {qf2_next_target.shape},next_state_log_pi={next_state_log_pi.shape}")
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                #print(f"rew {rew.shape} dones.flatten() {dones.shape}, min_qf_next_target={min_qf_next_target.shape}")
                next_q_value = rew[:,-1] + (1 - dones[:,-1]) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = trans_qf1(obs, action_target).reshape(-1)
            qf2_a_values = trans_qf2(obs, action_target).reshape(-1)

            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            trans_critic_RL_loss = qf1_loss + qf2_loss
            
            trans_q_optimizer.zero_grad()
            trans_critic_RL_loss.backward()
            trans_q_optimizer.step()
            
            logger.add_scalar("Trans/Actor_RL_loss", trans_actor_RL_loss.item(), global_step)
            logger.add_scalar("Trans/Critic_RL_loss", trans_critic_RL_loss.item(), global_step)
        
        if trans_global_update % args.target_network_frequency == 0:
            for param, target_param in zip(trans_qf1.parameters(), trans_qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(trans_qf2.parameters(), trans_qf2_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        
        
    duration = time.time() - start_time
    print(f"Transformer training completed in {duration:.2f} seconds.")
    
    
    
        
        
            
def evaluate_transformer():
    # evaluate
    rew_list = []
    
    trans_actor.eval()
    stime = time.perf_counter()
    
    eval_obs, _ = trans_eval_envs.reset()
    
    eval_observations = torch.empty((args.num_eval_envs, 0, envs.single_observation_space.shape[0])).to(device)   #n_e, 0, s_d
    eval_observations = torch.cat([eval_observations, eval_obs.unsqueeze(1)], dim=1)
    
    eval_metrics = defaultdict(list)
    num_episodes = 0
    for _ in range(args.num_eval_steps):  #num_eval_steps = 50
        
        if eval_observations.shape[1] > args.seq_len:
            eval_observations = eval_observations[:, -args.seq_len:,]
        
        
        with torch.no_grad():
            eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(trans_actor.get_eval_action(eval_observations))
            eval_observations = torch.cat([eval_observations, eval_obs.unsqueeze(1)], dim=1)
            rew_list.append(eval_rew.cpu().numpy().tolist())
            if "final_info" in eval_infos:
                mask = eval_infos["_final_info"]
                num_episodes += mask.sum()
                for k, v in eval_infos["final_info"]["episode"].items():
                    eval_metrics[k].append(v)
    eval_metrics_mean = {}
    for k, v in eval_metrics.items():
        mean = torch.stack(v).float().mean()
        eval_metrics_mean[k] = mean
        if logger is not None:
            logger.add_scalar(f"eval/{k}", mean, global_step)
    print(
        f"Trans success_once: {eval_metrics_mean['success_once']:.2f}, "
        f"Trans return: {eval_metrics_mean['return']:.2f}"
    )
    if logger is not None:
        eval_time = time.perf_counter() - stime
        cumulative_times["Trans_eval_time"] += eval_time
        logger.add_scalar("Trans/Trans_eval_time", eval_time, global_step)
        logger.add_scalar("Trans/Trans_eval_sr_once", eval_metrics_mean['success_once'], global_step)
        logger.add_scalar("Trans/Trans_eval_return", eval_metrics_mean['return'], global_step)        
        
        

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.grad_steps_per_iteration = int(args.training_freq * args.utd)
    args.steps_per_env = args.training_freq // args.num_envs
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"ECAI_CAMERA_READY_VEC/{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ####### Environment setup #######
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="gpu")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, human_render_camera_configs=dict(shader_pack="default"), **env_kwargs)
    trans_eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, human_render_camera_configs=dict(shader_pack="default"), **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
        trans_eval_envs = FlattenActionSpaceWrapper(trans_eval_envs)
    if args.capture_video or args.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.save_trajectory, save_video=args.capture_video, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    trans_eval_envs = ManiSkillVectorEnv(trans_eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    args.max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env) 
    
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=False)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=["sac", "walltime_efficient"]
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        actor.load_state_dict(ckpt['actor'])
        qf1.load_state_dict(ckpt['qf1'])
        qf2.load_state_dict(ckpt['qf2'])
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    trans_actor = Trans_Actor(envs, args).to(device)
    trans_qf1 = Trans_SoftQNetwork(envs, args).to(device)
    trans_qf2 = Trans_SoftQNetwork(envs, args).to(device)
    trans_qf1_target = Trans_SoftQNetwork(envs, args).to(device)
    trans_qf2_target = Trans_SoftQNetwork(envs, args).to(device)
    trans_qf1_target.load_state_dict(trans_qf1.state_dict())
    trans_qf2_target.load_state_dict(trans_qf2.state_dict())
    trans_q_optimizer = optim.Adam(list(trans_qf1.parameters()) + list(trans_qf2.parameters()), lr=args.q_lr)
    trans_actor_optimizer = optim.Adam(list(trans_actor.parameters()), lr=args.policy_lr)
    
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        env=envs,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        storage_device=torch.device(args.buffer_device),
        sample_device=device
    )


    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset(seed=args.seed) # in Gymnasium, seed is given to reset() instead of seed()
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    global_step = 0
    global_update = 0
    trans_global_update = 0
    learning_has_started = False

    global_steps_per_iteration = args.num_envs * (args.steps_per_env)
    pbar = tqdm.tqdm(range(args.total_timesteps))
    cumulative_times = defaultdict(float)
    rollout_positions = []      
    while global_step < args.total_timesteps:
        
        if args.eval_freq > 0 and (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq:
                evaluate_transformer()
                
                model_path = f"ECAI_CAMERA_READY_WEIGHTS_VEC/{args.env_id}/seed_{args.seed}|ckpt_{global_step}.pt"
                torch.save({
                    'trans_actor': trans_actor.state_dict(),
                    'trans_qf1': trans_qf1.state_dict(),
                    'trans_qf2': trans_qf1.state_dict(),
                    'qf1': qf1_target.state_dict(),
                    'qf2': qf2_target.state_dict(),
                    'log_alpha': log_alpha,
                }, model_path)
                print(f"trans saved to {model_path}")
        
        if args.eval_freq > 0 and (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq:
            # evaluate
            actor.eval()
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            positions = []
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_actions = actor.get_eval_action(eval_obs)
                    eval_actions = eval_actions.detach()
                    
                    eval_next_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(eval_actions)
                    real_next_obs = eval_next_obs
                    
                    stop_bootstrap = torch.zeros_like(eval_terminations, dtype=torch.bool)
                    pos = rb.add(eval_obs, real_next_obs, eval_actions, eval_rew, stop_bootstrap)
                    positions.append(pos)

                    # Обновляем obs на следующий шаг
                    eval_obs = eval_next_obs
                    
                    
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            eval_metrics_mean = {}
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                eval_metrics_mean[k] = mean
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                    
            #print(positions)
            rb.add_associated_reward(positions, eval_metrics['return'][0])
            rb.add_qv_estimates(positions)                    
                    
            pbar.set_description(
                f"success_once: {eval_metrics_mean['success_once']:.2f}, "
                f"return: {eval_metrics_mean['return']:.2f}"
            )
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.evaluate:
                break
            actor.train()

            # if args.save_model:
            #     model_path = f"runs/{run_name}/ckpt_{global_step}.pt"
            #     torch.save({
            #         'actor': actor.state_dict(),
            #         'qf1': qf1_target.state_dict(),
            #         'qf2': qf2_target.state_dict(),
            #         'log_alpha': log_alpha,
            #     }, model_path)
            #     print(f"model saved to {model_path}")
            
            train_transformer(loss_proportion='bc', positions=positions, ascent_on='transformer', batch_size=100, context=3, shuffle=True)
            trans_global_update += 1
            #evaluate_transformer()
        
        
        # Collect samples from environemnts
        rollout_time = time.perf_counter()
        
        
        for local_step in range(args.steps_per_env):
            
            global_step += 1 * args.num_envs

            # ALGO LOGIC: put action logic here
            if not learning_has_started:
                actions = torch.tensor(envs.action_space.sample(), dtype=torch.float32, device=device)
            else:
                actions, _, _ = actor.get_action(obs)
                actions = actions.detach()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            real_next_obs = next_obs.clone()
            if args.bootstrap_at_done == 'never':
                need_final_obs = torch.ones_like(terminations, dtype=torch.bool)
                stop_bootstrap = truncations | terminations # always stop bootstrap when episode ends
            else:
                if args.bootstrap_at_done == 'always':
                    need_final_obs = truncations | terminations # always need final obs when episode ends
                    stop_bootstrap = torch.zeros_like(terminations, dtype=torch.bool) # never stop bootstrap
                else: # bootstrap at truncated
                    need_final_obs = truncations & (~terminations) # only need final obs when truncated and not terminated
                    stop_bootstrap = terminations # only stop bootstrap when terminated, don't stop when truncated
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                real_next_obs[need_final_obs] = infos["final_observation"][need_final_obs]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

            pos = rb.add(obs, real_next_obs, actions, rewards, stop_bootstrap)
            #rollout_positions.append(pos)
            
            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        pbar.update(args.num_envs * args.steps_per_env)
        
        
        
        # ALGO LOGIC: training.
        if global_step < args.learning_starts:
            continue
        
        # if len(rollout_positions) >= 50:
        #     #print('rollout_positions')
        #     #print(rollout_positions)
            
        #     train_transformer(loss_proportion='bc', positions=rollout_positions, ascent_on='transformer', batch_size=100, context=3, shuffle=True)
            
        #     rollout_positions = []
            
        update_time = time.perf_counter()
        learning_has_started = True
        for local_update in range(args.grad_steps_per_iteration):
            global_update += 1
            data = rb.sample(args.batch_size)

            # update the value networks
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_obs)
                qf1_next_target = qf1_target(data.next_obs, next_state_actions)
                qf2_next_target = qf2_target(data.next_obs, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                # data.dones is "stop_bootstrap", which is computed earlier according to args.bootstrap_at_done

            qf1_a_values = qf1(data.obs, data.actions).view(-1)
            qf2_a_values = qf2(data.obs, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # update the policy network
            if global_update % args.policy_frequency == 0:  # TD 3 Delayed update support
                pi, log_pi, _ = actor.get_action(data.obs)
                qf1_pi = qf1(data.obs, pi)
                qf2_pi = qf2(data.obs, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(data.obs)
                    # if args.correct_alpha:
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                    # else:
                    #     alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()
                    # log_alpha has a legacy reason: https://github.com/rail-berkeley/softlearning/issues/136#issuecomment-619535356

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_update % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        # Log training-related data
        if (global_step - args.training_freq) // args.log_freq < global_step // args.log_freq:
            logger.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
            logger.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
            logger.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            logger.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            logger.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            logger.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            logger.add_scalar("losses/alpha", alpha, global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)
            logger.add_scalar("time/rollout_fps", global_steps_per_iteration / rollout_time, global_step)
            for k, v in cumulative_times.items():
                logger.add_scalar(f"time/total_{k}", v, global_step)
            logger.add_scalar("time/total_rollout+update_time", cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)
            if args.autotune:
                logger.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    if not args.evaluate and args.save_model:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save({
            'actor': actor.state_dict(),
            'qf1': qf1_target.state_dict(),
            'qf2': qf2_target.state_dict(),
            'log_alpha': log_alpha,
        }, model_path)
        print(f"model saved to {model_path}")
        writer.close()
    envs.close()
