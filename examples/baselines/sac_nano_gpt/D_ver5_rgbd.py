from collections import defaultdict
from dataclasses import dataclass
import os
import random
import time
from typing import Optional
import math
import tqdm

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
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



######### MY CUSTOM TOOLS FOR TRANSFORMER-BASED ONLINE MANISKILL #########
class ResetTracker:
    def __init__(self, n_envs, max_steps):
        self.n_envs = n_envs  # Количество сред
        self.max_steps = max_steps  # Максимальное количество шагов до удаления индекса
        self.step_counts = {}  # Хранит шаги после ресета {индекс среды: кол-во шагов}

    def add(self, reset_indices):
        """Регистрирует сброс для указанных индексов."""
        for index in reset_indices:
            if 0 <= index < self.n_envs: 
                self.step_counts[index] = 1  # Обнуляем счётчик шагов для этих индексов
            else:
                print('Error!')    

    def step(self):
        """Обновляет шаги для всех зарегистрированных индексов и удаляет просроченные."""
        to_remove = []
        for index in self.step_counts.keys():
            self.step_counts[index] += 1
            if self.step_counts[index] >= self.max_steps: # как только набираем необходимый контекст - удаляем индекс
                to_remove.append(index)
        for index in to_remove:
            del self.step_counts[index]

    def get_info(self, idx):
        return self.step_counts[idx]
    
    
    def get_active_indices(self):
        """Потом исп-ть эту ф-ю для нарезки данных. 
        Т.е. сначала срезаем всё, а затем на основе этих индексов корректируем срез"""
        return list(self.step_counts.keys())


def smart_slice(observations, context, tracker, element_space):
    """Возвращает срез observations с паддингом при недостатке шагов."""

    dummy_key = list(observations.keys())[0]
    obs_shape = observations[dummy_key].shape[:2]
    num_envs, seq_len = obs_shape

    seq_observation2RB = {} #DictArray(obs_shape, element_space, device=device)
    seq_next_observation2RB = {} #DictArray(obs_shape, element_space, device=device)

    for k in observations.keys():
        observations2RB = []
        next_observations2RB = []

        if seq_len >= context:
            for env_index in range(num_envs):
                if env_index in tracker.step_counts.keys(): # если эта среда нуждается в аккуратном срезе
                    steps_since_reset = tracker.get_info(env_index)  # узнаём сколько шагов простепали после ресета
                    padding_size = context - steps_since_reset       # считаем сколько шагов надо допадить

                    # padding = observations[k][env_index, -steps_since_reset, :].unsqueeze(0).repeat(padding_size, 1) # формируем паддинг
                    padding = observations[k][env_index, -steps_since_reset, :].unsqueeze(0)  # добавляем измерение последовательности
                    if padding.dim() == 2:
                        # Для тензора размерности [B, 1, D] (например, [4, 1, 25])
                        padding = padding.repeat(padding_size, 1)  # результат: [B, padding_size, D]
                    elif padding.dim() == 4:
                        # Для тензора размерности [B, 1, H, W, C] (например, [4, 1, 64, 64, 3])
                        padding = padding.repeat(padding_size, 1, 1, 1)  # результат: [B, padding_size, H, W, C]
                    else:
                        raise ValueError("Непредвиденное число размерностей у padding тензора")

                    valid_states = observations[k][env_index, -steps_since_reset:, :]
                    next_obs = torch.cat([padding, valid_states], dim=0)  # работает если после ресета прошло мин 2 степа
                    obs = torch.cat([ padding[0].unsqueeze(0), padding, valid_states[:-1,:] ], dim=0)# работает если после ресета прошло мин 2 степа
                    next_observations2RB.append(next_obs)
                    observations2RB.append(obs)
                elif env_index not in tracker.step_counts.keys():
                    next_obs = observations[k][env_index, -context:, :] # всегда cont, s_d
                    obs = torch.cat([ next_obs[0,:].unsqueeze(0), next_obs[:-1,:] ])   # а точно ли я должен next_obs[0,:] добавлять или надо просто другой срез сделать? 
                    next_observations2RB.append(next_obs)
                    observations2RB.append(obs)
            
            seq_observation2RB[k], seq_next_observation2RB[k]  = torch.stack(observations2RB), torch.stack(next_observations2RB)
        else:                
            padding_size = context - seq_len
            
            padding = observations[k][:, 0, :].unsqueeze(1)  # добавляем измерение последовательности

            if padding.dim() == 3:
                # Для тензора размерности [B, 1, D] (например, [4, 1, 25])
                padding = padding.repeat(1, padding_size, 1)  # результат: [B, padding_size, D]
            elif padding.dim() == 5:
                # Для тензора размерности [B, 1, H, W, C] (например, [4, 1, 64, 64, 3])
                padding = padding.repeat(1, padding_size, 1, 1, 1)  # результат: [B, padding_size, H, W, C]
            else:
                raise ValueError("Непредвиденное число размерностей у padding тензора")
            
            seq_next_observation2RB[k] = torch.cat([padding, observations[k]], dim=1)

            seq_observation2RB[k] = torch.cat([
                    observations[k][:, 0, :].unsqueeze(1),  # начальное состояние
                    padding,
                    observations[k][:,:-1, :]
                ], dim=1)
            

    return seq_observation2RB, seq_next_observation2RB






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
    save_model: bool = False
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
    obs_mode: str = "rgb"
    """the observation mode to use"""
    include_state: bool = True
    """whether to include the state in the observation"""
    render_mode: str = "all"
    """the environment rendering mode"""
    env_vectorization: str = "gpu"
    """the type of environment vectorization to use"""
    num_envs: int = 50
    """the number of parallel environments"""
    num_eval_envs: int = 16
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
    eval_freq: int = 50000
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""

    # Algorithm specific arguments
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    context: int = 10
    """transformer context"""
    buffer_size: int = 1_000_000
    """the replay memory buffer size"""
    buffer_device: str = "cuda"
    """where the replay buffer is stored. Can be 'cpu' or 'cuda' for GPU"""
    gamma: float = 0.8
    """the discount factor gamma"""
    tau: float = 0.01
    """target smoothing coefficient"""
    batch_size: int = 512
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
    utd: float = 0.25
    """update to data ratio"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    bootstrap_at_done: str = "always"
    """the bootstrap method to use when a done signal is received. Can be 'always' or 'never'"""
    camera_width: Optional[int] = None
    """the width of the camera image. If none it will use the default the environment specifies"""
    camera_height: Optional[int] = None
    """the height of the camera image. If none it will use the default the environment specifies."""

    # to be filled in runtime
    grad_steps_per_iteration: int = 0 # = training_freq*utd=32 by default 
    """the number of gradient updates per iteration"""
    steps_per_env: int = 0
    """the number of steps each parallel env takes per iteration"""
    
    
    ##### SPECIAL ARGS FOR TRANSFORMER  ##########
    
    use_gates: bool = False
    """use gatings instead skip connection in transformer block"""
    n_embd: int = 256
    """inner transformer dimention"""
    n_layer: int = 1
    n_head: int = 2
    dropout: float = 0.0
    seq_len: int = 10
    bias: bool = True

class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (torch.float32 if v.dtype in (np.float32, np.float64) else
                            torch.uint8 if v.dtype == np.uint8 else
                            torch.int16 if v.dtype == np.int16 else
                            torch.int32 if v.dtype == np.int32 else
                            v.dtype)
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        else:
            for k, v in value.items():
                self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k,v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)



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

#######################################################################################################################################
##########################################################  Gatings  #################################################################

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

#######################################################################################################################################
######################################################## nano gpt modification ########################################################

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


@dataclass
class ReplayBufferSample:
    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
class ReplayBuffer:
    def __init__(self, env, num_envs: int, context: int, buffer_size: int, storage_device: torch.device, sample_device: torch.device):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device
        self.per_env_buffer_size = buffer_size // num_envs
        # note 128x128x3 RGB data with replay buffer size 100_000 takes up around 4.7GB of GPU memory
        # 32 parallel envs with rendering uses up around 2.2GB of GPU memory.
        self.obs = DictArray((self.per_env_buffer_size, num_envs, context), env.single_observation_space, device=storage_device)
        # TODO (stao): optimize final observation storage
        self.next_obs = DictArray((self.per_env_buffer_size, num_envs, context), env.single_observation_space, device=storage_device)
        self.actions = torch.zeros((self.per_env_buffer_size, num_envs) + env.single_action_space.shape, device=storage_device)
        self.logprobs = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        self.rewards = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        self.dones = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        self.values = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)

    def add(self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor):
        if self.storage_device == torch.device("cpu"):
            obs = {k: v.cpu() for k, v in obs.items()}
            next_obs = {k: v.cpu() for k, v in next_obs.items()}
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()

        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs

        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0
    def sample(self, batch_size: int):
        if self.full:
            batch_inds = torch.randint(0, self.per_env_buffer_size, size=(batch_size, ))
        else:
            batch_inds = torch.randint(0, self.pos, size=(batch_size, ))
        env_inds = torch.randint(0, self.num_envs, size=(batch_size, ))
        obs_sample = self.obs[batch_inds, env_inds]
        next_obs_sample = self.next_obs[batch_inds, env_inds]
        obs_sample = {k: v.to(self.sample_device) for k, v in obs_sample.items()}
        next_obs_sample = {k: v.to(self.sample_device) for k, v in next_obs_sample.items()}
        return ReplayBufferSample(
            obs=obs_sample,
            next_obs=next_obs_sample,
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device)
        )

# ALGO LOGIC: initialize agent here:
class PlainConv(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_dim=256,
                 pool_feature_map=False,
                 last_act=True, # True for ConvBody, False for CNN
                 image_size=[128, 128]
                 ):
        super().__init__()
        # assume input image size is 128x128 or 64x64

        self.out_dim = out_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4) if image_size[0] == 128 and image_size[1] == 128 else nn.MaxPool2d(2, 2),  # [32, 32]
            nn.Conv2d(16, 32, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [16, 16]
            nn.Conv2d(32, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [8, 8]
            nn.Conv2d(64, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [4, 4]
            nn.Conv2d(64, 64, 1, padding=0, bias=True), nn.ReLU(inplace=True),
        )

        if pool_feature_map:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = make_mlp(128, [out_dim], last_act=last_act)
        else:
            self.pool = None
            self.fc = make_mlp(64 * 4 * 4, [out_dim], last_act=last_act)

        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image):
        x = self.cnn(image)
        if self.pool is not None:
            x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

# class Encoder(nn.Module):
#     def __init__(self, sample_obs):
#         super().__init__()

#         extractors = {}

#         self.out_features = 0
#         feature_size = 256
#         in_channels=sample_obs["rgb"].shape[-1]
#         image_size=(sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])


#         # here we use a NatureCNN architecture to process images, but any architecture is permissble here
#         cnn = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=in_channels,
#                 out_channels=32,
#                 kernel_size=8,
#                 stride=4,
#                 padding=0,
#             ),
#             nn.ReLU(),
#             nn.Conv2d(
#                 in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
#             ),
#             nn.ReLU(),
#             nn.Conv2d(
#                 in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
#             ),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         # to easily figure out the dimensions after flattening, we pass a test tensor
#         with torch.no_grad():
#             n_flatten = cnn(sample_obs["rgb"].float().permute(0,3,1,2).cpu()).shape[1]
#             fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
#         extractors["rgb"] = nn.Sequential(cnn, fc)
#         self.out_features += feature_size
#         self.extractors = nn.ModuleDict(extractors)

#     def forward(self, observations) -> torch.Tensor:
#         encoded_tensor_list = []
#         # self.extractors contain nn.Modules that do all the processing.
#         for key, extractor in self.extractors.items():
#             obs = observations[key]
#             if key == "rgb":
#                 obs = obs.float().permute(0,3,1,2)
#                 obs = obs / 255
#             encoded_tensor_list.append(extractor(obs))
#         return torch.cat(encoded_tensor_list, dim=1)

# class EncoderObsWrapper(nn.Module):
#     def __init__(self, encoder):
#         super().__init__()
#         self.encoder = encoder

#     def forward(self, obs):
#         if "rgb" in obs.keys():
#             rgb = obs['rgb'].float() / 255.0 # (B, H, W, 3*k)
#         if "depth" in obs.keys():
#             depth = obs['depth'].float() # (B, H, W, 1*k)
#         if "rgb" in obs.keys() and "depth" in obs.keys():
#             img = torch.cat([rgb, depth], dim=3) # (B, H, W, C)
#         elif "rgb" in obs.keys():
#             img = rgb
#         elif "depth" in obs.keys():
#             img = depth
#         else:
#             raise ValueError(f"Observation dict must contain 'rgb' or 'depth'")
#         print(img.shape)
#         img = img.permute(0, 3, 1, 2) # (B, C, H, W)
#         return self.encoder(img)

class EncoderObsWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, obs):
        # Обработка входа: берём либо 'rgb', либо 'depth', либо их сочетание.
        if "rgb" in obs.keys():
            # Предполагаем, что изображения в rgb могут быть в формате [B, seq_len, H, W, C] или [B, H, W, C]
            rgb = obs['rgb'].float() / 255.0
        if "depth" in obs.keys():
            depth = obs['depth'].float()
        
        if "rgb" in obs.keys() and "depth" in obs.keys():
            img = torch.cat([rgb, depth], dim=-1)  # Конкатенация по последней размерности (каналы)
        elif "rgb" in obs.keys():
            img = rgb
        elif "depth" in obs.keys():
            img = depth
        else:
            raise ValueError("Observation dict must contain 'rgb' or 'depth'")
        
        # print("Input image shape:", img.shape)

        # Определяем, является ли вход последовательностью изображений.
        # Если изображения имеют размерность 5, например, (B, seq_len, H, W, C) или (B, seq_len, C, H, W)
        if img.ndim == 5:
            # Допустим, исходно изображения имеют форму (B, seq_len, H, W, C)
            # Если у вас формат (B, seq_len, H, W, C), то сначала переставим оси так, чтобы получить (B, seq_len, C, H, W):
            if img.shape[-1] not in {1, 3, 4}:  # проверка: число каналов должно быть 1, 3 или 4
                raise ValueError("Ожидаем, что последний размер соответствует числу каналов")
            img = img.permute(0, 1, 4, 2, 3)  # (B, seq_len, C, H, W)
            
            B, seq_len, C, H, W = img.shape
            # Объединяем размеры batch и sequence, чтобы энкодер получил 4D-тензор
            img = img.reshape(B * seq_len, C, H, W)
            
            # Пропускаем через энкодер
            features = self.encoder(img)  # ожидается форма (B * seq_len, feature_dim)

            # Если нужно вернуть обратно информацию по последовательностям:
            feature_dim = features.shape[-1]
            features = features.view(B, seq_len, feature_dim)
            return features
        else:
            # Если тензор уже имеет 4 размерности (B, H, W, C)
            img = img.permute(0, 3, 1, 2)  # (B, C, H, W)
            return self.encoder(img)


def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    '''
    Q-network for Transformer-based maniskill tasks
    '''
    def __init__(self, envs, args, encoder: EncoderObsWrapper):
        super().__init__()
        # self.encoder = nn.Linear(np.array(env.single_observation_space.shape).prod(), args.n_embd)
        # self.net = nn.Sequential(
        #     nn.Linear(args.n_embd + np.prod(env.single_action_space.shape), 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1),
        # )
        self.encoder = encoder
        action_dim = np.prod(envs.single_action_space.shape)
        self.state_dim = envs.single_observation_space['state'].shape[0] if 'state' in envs.single_observation_space.keys() else 0
        self.mlp = make_mlp(args.n_embd+action_dim, [512, 256, 1], last_act=False)
        self.transformer = GPT(args)

    def forward(self, obs, action, visual_feature=None, detach_encoder=False):
        if visual_feature is None:
            visual_feature = self.encoder(obs)
        if detach_encoder:
            visual_feature = visual_feature.detach()
        if self.state_dim != 0:
            mixed_feature = torch.cat([visual_feature, obs["state"]], dim=1)
            belief_state = self.transformer(mixed_feature)[:, -1, :]   # x = (batch,n_embd)
            x = torch.cat([belief_state, action], dim=1)
        else:
            belief_state = self.transformer(visual_feature)[:, -1, :]   # x = (batch,n_embd)
            x = torch.cat([belief_state, action], dim=1)
        return self.mlp(x)

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, envs, args, sample_obs):
        super().__init__()
        action_dim = np.prod(envs.single_action_space.shape)
        self.state_dim = envs.single_observation_space['state'].shape[0] if 'state' in envs.single_observation_space.keys() else 0
        # print(envs.single_observation_space)
        # count number of channels and image size
        in_channels = 0
        if "rgb" in sample_obs:
            in_channels += sample_obs["rgb"].shape[-1]
            image_size = sample_obs["rgb"].shape[1:3]
        if "depth" in sample_obs:
            in_channels += sample_obs["depth"].shape[-1]
            image_size = sample_obs["depth"].shape[1:3]

        self.encoder = EncoderObsWrapper(
            PlainConv(in_channels=in_channels, out_dim=256, image_size=image_size) # assume image is 64x64
        )
        self.mlp = make_mlp(args.n_embd+self.state_dim, [512, 256], last_act=True)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        # action rescaling
        self.action_scale = torch.FloatTensor((envs.single_action_space.high - envs.single_action_space.low) / 2.0)
        self.action_bias = torch.FloatTensor((envs.single_action_space.high + envs.single_action_space.low) / 2.0)
        self.transformer = GPT(args)

    def get_feature(self, obs, detach_encoder=False):
        # print(obs.keys())
        k1 = list(obs.keys())[-1]
        # print("rgb" in obs)
        # print(k1 in obs.keys())
        visual_feature = self.encoder(obs)
        if detach_encoder:
            visual_feature = visual_feature.detach()
        if self.state_dim != 0:
            # print(visual_feature.shape, obs["state"].shape)
            mixed_feature = torch.cat([visual_feature, obs["state"]], dim=1)
            # print(f'{mixed_feature.shape=}')
            x = self.transformer(mixed_feature)[:, -1, :]   # x = (batch,n_embd)
            # x = torch.cat([visual_feature, obs['state']], dim=1)
        else:
            # print(f'{visual_feature.shape=}')
            x = self.transformer(visual_feature)[:, -1, :]   # x = (batch,n_embd)
            # x = visual_feature
        return self.mlp(x), visual_feature

    def forward(self, obs, detach_encoder=False):
        x, visual_feature = self.get_feature(obs, detach_encoder)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std, visual_feature

    def get_eval_action(self, obs):
        mean, log_std, _ = self(obs)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_action(self, obs, detach_encoder=False):
        mean, log_std, visual_feature = self(obs, detach_encoder)
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
        return action, log_prob, mean, visual_feature

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




if __name__ == "__main__":
    args = tyro.cli(Args)
    args.grad_steps_per_iteration = int(args.training_freq * args.utd) #32
    args.steps_per_env = args.training_freq // args.num_envs           #64/32=2
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ####### Environment setup #######
    env_kwargs = dict(obs_mode=args.obs_mode, render_mode=args.render_mode, sim_backend="gpu", sensor_configs=dict())
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    if args.camera_width is not None:
        # this overrides every sensor used for observation generation
        env_kwargs["sensor_configs"]["width"] = args.camera_width
    if args.camera_height is not None:
        env_kwargs["sensor_configs"]["height"] = args.camera_height
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, human_render_camera_configs=dict(shader_pack="default"), **env_kwargs)

    # rgbd obs mode returns a dict of data, we flatten it so there is just a rgbd key and state key
    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=args.include_state)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=args.include_state)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
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
                tags=["sac_gpt", "walltime_efficient"]
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")


    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        env=envs,
        num_envs=args.num_envs,
        context=args.seq_len,
        buffer_size=args.buffer_size,
        storage_device=torch.device(args.buffer_device),
        sample_device=device
    )

    # max_action = float(envs.single_action_space.high[0])

    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset(seed=args.seed) # in Gymnasium, seed is given to reset() instead of seed()
    eval_obs, _ = eval_envs.reset(seed=args.seed)



    # if args.include_state:
    #     args.n_embd = # visual feature dim + state dim


    # architecture is all actor, q-networks share the same vision encoder. Output of encoder is concatenates with any state data followed by separate MLPs.
    actor = Actor(envs, args, sample_obs=obs).to(device)
    qf1 = SoftQNetwork(envs, args, actor.encoder).to(device)
    qf2 = SoftQNetwork(envs, args, actor.encoder).to(device)
    qf1_target = SoftQNetwork(envs, args, actor.encoder).to(device)
    qf2_target = SoftQNetwork(envs, args, actor.encoder).to(device)
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        actor.load_state_dict(ckpt['actor'])
        qf1.load_state_dict(ckpt['qf1'])
        qf2.load_state_dict(ckpt['qf2'])
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.mlp.parameters()) +
        list(qf2.mlp.parameters()) +
        list(qf1.encoder.parameters()),
        lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha


    global_step = 0
    global_update = 0
    learning_has_started = False

    global_steps_per_iteration = args.num_envs * (args.steps_per_env) # 32*2=64
    pbar = tqdm.tqdm(range(args.total_timesteps))
    cumulative_times = defaultdict(float)


    
    tracker = ResetTracker(n_envs=args.num_envs, max_steps=args.seq_len)


    global_observations = DictArray((args.num_envs, 0), envs.single_observation_space, device=device)

    for k in obs.keys():
        # print(k, global_observations[k].shape, obs[k].shape)
        global_observations[k] = torch.cat([global_observations[k], obs[k].unsqueeze(1)], dim=1)
    # print(global_observations.keys())

    # global_observations = torch.empty((args.num_envs, 0, envs.single_observation_space.shape[0])).to(device)   #n_e, 0, s_d
    # global_observations = torch.cat([global_observations, obs.unsqueeze(1)], dim=1)

    while global_step < args.total_timesteps:
        
    #↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    #↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓     EVALUATION     ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    #↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓        
            
        if args.eval_freq > 0 and (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq:
            # evaluate
            actor.eval()
            stime = time.perf_counter()
            
            eval_obs, _ = eval_envs.reset()

            eval_observations = DictArray((args.num_eval_envs, 0), eval_envs.single_observation_space, device=device)
            for k in eval_obs.keys():
                eval_observations[k] = torch.cat([eval_observations[k], eval_obs[k].unsqueeze(1)], dim=1)

            # eval_observations = torch.empty((args.num_eval_envs, 0, envs.single_observation_space.shape[0])).to(device)   #n_e, 0, s_d
            # eval_observations = torch.cat([eval_observations, eval_obs.unsqueeze(1)], dim=1)
            
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for eval_step in range(args.num_eval_steps):  #num_eval_steps = 50
                print(f'{eval_step=}')

                if eval_observations.shape[1] > args.seq_len:
                    eval_observations = eval_observations[:, -args.seq_len:,]
                
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(actor.get_eval_action(eval_observations))

                for k in eval_obs.keys():
                    eval_observations[k] = torch.cat([eval_observations[k], eval_obs[k].unsqueeze(1)], dim=1)

                    # eval_observations = torch.cat([eval_observations, eval_obs.unsqueeze(1)], dim=1)
                    
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

        
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓     ROLLOUT     ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓  
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        
        # Collect samples from environemnts
        rollout_time = time.perf_counter()
        for local_step in range(args.steps_per_env):
            global_step += 1 * args.num_envs

            if global_observations.shape[1] > args.seq_len:
                global_observations = global_observations[:, -args.seq_len-1:,]
            obs2act = smart_slice(global_observations, args.seq_len, tracker, envs.single_observation_space)[1]
            
            if not learning_has_started:
                actions = torch.tensor(envs.action_space.sample(), dtype=torch.float32, device=device)
            else:
                actions, _, _, _ = actor.get_action(obs2act)
                actions = actions.detach()

            
            
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            
            # global_observations = torch.cat([global_observations, next_obs.unsqueeze(1)], dim=1)

            for k in global_observations.keys():
                global_observations[k] = torch.cat([global_observations[k], next_obs[k].unsqueeze(1)], dim=1)
            tracker.step() 
            
            
            # real_next_obs = next_obs.clone()
            real_next_obs = {k:v.clone() for k, v in next_obs.items()}

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
                for k in real_next_obs.keys():
                    real_next_obs[k][need_final_obs] = infos["final_observation"][k][need_final_obs].clone()
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
            
            
            real_global_observations = {k:v.clone() for k, v in global_observations.data.items()} # global_observations.clone()
            # print(real_global_observations)
            for k in real_global_observations.keys():
                real_global_observations[k][:,-1,:] = real_next_obs[k]
            obs2RB, n_obs2RB = smart_slice(real_global_observations, args.seq_len, tracker, envs.single_observation_space)
            
            rb.add(obs2RB, n_obs2RB, actions, rewards, stop_bootstrap)   # RB* <-- (ne,cont,sd), (ne,cont,sd), (ne,ad), (ne), (ne) 
            
            
            if 'episode' in infos and infos['episode']['success_once'].any():
                success_indices = torch.where(infos['episode']['success_once'])[0].tolist() # выводим индексы тех сред где случился success
                reseted_obs, _ = envs.reset(options={"env_idx":success_indices}) # обновляем только те среды, в который словили sr_once. при этом остальныве среды остаются неизменными
                tracker.add(success_indices)
                global_observations[:,-1,:] = reseted_obs
               

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        pbar.update(args.num_envs * args.steps_per_env)

        # assert 0
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓     TRAIN     ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓   
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
 
        # ALGO LOGIC: training.
        if global_step < args.learning_starts:
            continue

        update_time = time.perf_counter()
        learning_has_started = True
        for local_update in range(args.grad_steps_per_iteration):
            global_update += 1
            data = rb.sample(args.batch_size)

            # update the value networks
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _, visual_feature = actor.get_action(data.next_obs)
                qf1_next_target = qf1_target(data.next_obs, next_state_actions, visual_feature)
                qf2_next_target = qf2_target(data.next_obs, next_state_actions, visual_feature)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                # data.dones is "stop_bootstrap", which is computed earlier according to args.bootstrap_at_done
            visual_feature = actor.encoder(data.obs)
            qf1_a_values = qf1(data.obs, data.actions, visual_feature).view(-1)
            qf2_a_values = qf2(data.obs, data.actions, visual_feature).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # update the policy network
            if global_update % args.policy_frequency == 0:  # TD 3 Delayed update support
                pi, log_pi, _, visual_feature = actor.get_action(data.obs)
                qf1_pi = qf1(data.obs, pi, visual_feature, detach_encoder=True)
                qf2_pi = qf2(data.obs, pi, visual_feature, detach_encoder=True)
                min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _, _ = actor.get_action(data.obs)
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
