
from collections import defaultdict
from dataclasses import dataclass
import os
import random
import time
from typing import Optional
import math
import tqdm
import json


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


def smart_slice(observations, context, elapsed_steps, for_rb=True):
    
    num_envs = observations.shape[0]
        
    observations2RB = []
    next_observations2RB = []
    
    if for_rb: # если режем для rb (готовим и obs и next_obs)
        for env_index in range(num_envs): # пробегаем по средам
            
            steps_since_reset = elapsed_steps[env_index] # сколько шагов настепали с последнего ресета 
            if steps_since_reset >= context:  # этого условия достаточно что бы без пад-гов сделать s и s'
                obs = observations[env_index, -context-1:-1, ]
                next_obs = observations[env_index, -context:, ]
            
            elif 0 < steps_since_reset < context:                
                padding_size = context - steps_since_reset 
                
                if len(observations.shape) > 4:
                    padding = observations[env_index, -steps_since_reset-1, ].unsqueeze(0).repeat(padding_size, 1, 1, 1 )
                else:
                    padding = observations[env_index, -steps_since_reset-1, ].unsqueeze(0).repeat(padding_size, 1 )
                    
                 
                valid_states = observations[env_index, -steps_since_reset-1:, ]
                intermediate_obs = torch.cat([padding, valid_states], dim=0) 
                obs = intermediate_obs[ -context-1:-1, ]
                next_obs = intermediate_obs[ -context:, ]
            
            elif  steps_since_reset == 0:  # это обманка, на самом деле мы не начали с нового сост-я а видим 51й кадр
                obs = observations[env_index, -context-1:-1, ]
                next_obs = observations[env_index, -context:, ]
                
            observations2RB.append(obs)
            next_observations2RB.append(next_obs)  
            
        return torch.stack(observations2RB), torch.stack(next_observations2RB)
                
    if not for_rb:
        '''
        Значит мы на этапе беганья по среде (TRAIN)
        
        если elapsed steps среды =0, то значит среда только что ресетнулась, мы видим первый стетйт но действией еще не делали 
        '''
        
        for env_index in range(num_envs): # пробегаем по средам
            
            steps_since_reset = elapsed_steps[env_index]  # м.б. =0,1,2....
            if steps_since_reset >= context-1:                  # самый позитивый сценарий, просто нарезаем и не паримся
                next_obs = observations[env_index, -context:, :]
                
            elif steps_since_reset < context-1: 
                #print(f"observations {observations.shape}")
                
                padding_size = context - steps_since_reset - 1
                #print(f"padding_size {padding_size}")
                
                if len(observations.shape) > 4:
                    #print(f"Long obs { observations.shape }")
                    padding = observations[env_index, -steps_since_reset-1, ].unsqueeze(0).repeat(padding_size, 1, 1, 1 )
                else:
                    #print(f"Short obs { observations.shape }")
                    padding = observations[env_index, -steps_since_reset-1, ].unsqueeze(0).repeat(padding_size, 1 )
                #print(f"padding {padding.shape}")
                
                valid_states = observations[env_index, -steps_since_reset-1:, ]
                #print(f"valid_states  {valid_states.shape}")

                next_obs = torch.cat([padding, valid_states], dim=0)
            
            # elif steps_since_reset == 0: # это я добавил недавно: случай когда мы только ресетнули среду
            #     padding_size = context - 1 # т.е. допаживаем оставшийся контекст просто тем самым нулевым стейтом коорый у нас есть
            #     padding = observations[env_index, -1, :].unsqueeze(0).repeat(padding_size, 1)  # берём текущее состояние и просто его растягиваем 
            #     valid_states = observations[env_index, -1, :] # одно валидное состояние, которое есть сейчас (тот самый первый степ)
            #     next_obs = torch.cat([padding, valid_states], dim=0)
                
            next_observations2RB.append(next_obs)
        
        return torch.stack(next_observations2RB)






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
    obs_mode: str = "rgb"
    """the observation mode to use"""
    include_state: bool = True
    """whether to include the state in the observation"""
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
    render_mode: str = "all"
    """the environment rendering mode"""
    

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
    batch_size: int = 820 #1024
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
    n_embd: int = 227#256
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

    def __init__(self, config, input_dim):
        super().__init__()
        #assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(input_dim, 3 * input_dim, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(input_dim, input_dim, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = input_dim
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

    def __init__(self, config, input_dim):
        super().__init__()
        self.c_fc    = nn.Linear(input_dim, 4 * input_dim, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * input_dim, input_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config, inner_size):
        super().__init__()
        
        encoder = EncoderObsWrapper
        
        
        self.ln_1 = LayerNorm(inner_size, bias=config.bias)
        self.attn = CausalSelfAttention(config, inner_size)
        self.ln_2 = LayerNorm(inner_size, bias=config.bias)
        self.mlp = MLP(config, inner_size)

        if config.use_gates:
            self.skip_fn_1 = GRUGate(inner_size, 2.0)
            self.skip_fn_2 = GRUGate(inner_size, 2.0)
        else:
            self.skip_fn_1 = lambda x, y: x + y
            self.skip_fn_2 = lambda x, y: x + y


    def forward(self, x):

        x = self.skip_fn_1(x, self.attn(self.ln_1(x)))
        x = self.skip_fn_2(x, self.mlp(self.ln_2(x)))

        
        return x

class GPT(nn.Module):

    def __init__(self, config, inner_size):
        super().__init__()

        self.config = config
        self.pos_embedding = nn.Embedding(config.max_episode_steps, inner_size)

        self.transformer_layers = nn.ModuleList([Block(config, inner_size) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(inner_size, bias=config.bias)
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
        
        #print(f"x: {x.shape}") # 32 256
        #print(f"pe: {pos_emb.shape}")  # 256 256
        
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
        self.obs = DictArray((self.per_env_buffer_size, num_envs, context), env.single_observation_space, device=storage_device)
        self.next_obs = DictArray((self.per_env_buffer_size, num_envs, context), env.single_observation_space, device=storage_device)
        self.actions = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_action_space.shape).to(storage_device)
        self.logprobs = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)
        self.rewards = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)
        self.dones = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)
        self.values = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device) 

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
    '''
    Conv Net constructor
    '''
    def __init__(self,
                 in_channels=3,
                 out_dim=227, #256,
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

class EncoderObsWrapper(nn.Module):
    '''
    Preparation module before applying CNN
    '''
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, obs):
        if "rgb" in obs:
            rgb = obs['rgb'].float() / 255.0 # (B, H, W, 3*k)
        if "depth" in obs:
            depth = obs['depth'].float() # (B, H, W, 1*k)
        if "rgb" and "depth" in obs:
            img = torch.cat([rgb, depth], dim=3) # (B, H, W, C)
        elif "rgb" in obs:
            img = rgb
        elif "depth" in obs:
            img = depth
        else:
            raise ValueError(f"Observation dict must contain 'rgb' or 'depth'")
        
        #print(img.shape)
        if len(img.shape) == 5: # we are on evaluation step
            n_e, cont, h, w, c = img.shape
            img = img.reshape(n_e*cont, h, w, c)
            img = img.permute(0, 3, 1, 2) # (B, C, H, W)
            img = self.encoder(img)
            img = img.reshape(n_e, cont, self.encoder.out_dim)
            
        else:                   # we are on train step
            bs, n_e, cont, h, w, c = img.shape
            img = img.reshape(bs*n_e*cont, h, w, c)
            img = img.permute(0, 3, 1, 2) # (B, C, H, W)
            img = self.encoder(img)
            img = img.reshape(bs, n_e, cont, self.encoder.out_dim)
        
        
        return img

def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)

class Actor(nn.Module):
    def __init__(self, env, args, sample_obs):
        super().__init__()
        action_dim = np.prod(envs.single_action_space.shape)
        self.state_dim = envs.single_observation_space['state'].shape[0] if 'state' in envs.single_observation_space.keys() else 0
        print(f"actor sd = {self.state_dim}")
        # count number of channels and image size
        in_channels = 0
        if "rgb" in sample_obs:
            in_channels += sample_obs["rgb"].shape[-1]
            image_size = sample_obs["rgb"].shape[1:3]
        if "depth" in sample_obs:
            in_channels += sample_obs["depth"].shape[-1]
            image_size = sample_obs["depth"].shape[1:3]

        self.encoder = EncoderObsWrapper(
            PlainConv(in_channels=in_channels, out_dim=227, image_size=image_size) # assume image is 64x64
        )
        inner_size = self.encoder.encoder.out_dim+self.state_dim
        self.fc_mean = nn.Linear(inner_size, action_dim)
        self.fc_logstd = nn.Linear(inner_size, action_dim)
        self.action_scale = torch.FloatTensor((envs.single_action_space.high - envs.single_action_space.low) / 2.0)
        self.action_bias = torch.FloatTensor((envs.single_action_space.high + envs.single_action_space.low) / 2.0)
        
        self.transformer = GPT(args, inner_size)

    def get_feature(self, obs, detach_encoder=False):
        #print(f"x before cnn {obs[args.obs_mode].shape} and {obs['state'].shape}")
        visual_feature = self.encoder(obs)
        #print(f"x before cat with state {visual_feature.shape}")
        if detach_encoder:
            visual_feature = visual_feature.detach()
        x = torch.cat([visual_feature, obs['state']], dim=-1)
        #print(f"x after cat with state {x.shape}")
    
        return self.transformer(x)[:,-1,:], visual_feature
    
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


class SoftQNetwork(nn.Module):
    '''
    Q-network for Transformer-based maniskill tasks
    '''
    def __init__(self, env, args, encoder: EncoderObsWrapper):
        super().__init__()
        self.encoder = encoder
        action_dim = np.prod(env.single_action_space.shape)
        self.state_dim = env.single_observation_space['state'].shape[0] if 'state' in env.single_observation_space.keys() else 0
        print(f"q_net sd = {self.state_dim}")
        inner_size = encoder.encoder.out_dim+self.state_dim
        
        self.transformer = GPT(args, inner_size)
        
        self.net = nn.Sequential(
            nn.Linear(encoder.encoder.out_dim+action_dim+self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs, action, visual_feature=None, detach_encoder=False):
        if visual_feature is None:
            visual_feature = self.encoder(obs) # img -> vec
        if detach_encoder:
            visual_feature = visual_feature.detach()
        if self.state_dim != 0:
            trans_inp = torch.cat([visual_feature, obs["state"]], dim=-1)
            trans_out = self.transformer(trans_inp)[:, -1, :]
        else:
            trans_out = self.transformer(visual_feature)[:, -1, :]
        x = torch.cat([trans_out, action], dim=-1) 
        
        return self.net(x)

LOG_STD_MAX = 2
LOG_STD_MIN = -5




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
        run_name = f"[LAST_PAD]{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    
    print(f"partial_reset {args.partial_reset}")
    print(f"eval partial_reset {args.eval_partial_reset}")
    
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
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        env=envs,
        num_envs=args.num_envs,
        context=args.seq_len,
        buffer_size=args.buffer_size,
        storage_device=torch.device(args.buffer_device),
        sample_device=device
    )
    
    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset(seed=args.seed) # in Gymnasium, seed is given to reset() instead of seed()
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    
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
        list(qf1.transformer.parameters()) +
        list(qf2.transformer.parameters()) +
        list(qf1.net.parameters()) +
        list(qf2.net.parameters()) +
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

    


    # TRY NOT TO MODIFY: start the game
    global_step = 0
    global_update = 0
    learning_has_started = False

    global_steps_per_iteration = args.num_envs * (args.steps_per_env) # 32*2=64
    pbar = tqdm.tqdm(range(args.total_timesteps))
    cumulative_times = defaultdict(float)

    
    h,w,c = envs.single_observation_space[args.obs_mode].shape
    global_img_observations = torch.empty((args.num_envs, 0, h, w, c)).to(device)   #n_e, 0, s_d
    global_img_observations = torch.cat([global_img_observations, obs[args.obs_mode].unsqueeze(1)], dim=1)
    
    global_vec_observations = torch.empty((args.num_envs, 0, envs.single_observation_space['state'].shape[0])).to(device)   #n_e, 0, s_d
    global_vec_observations = torch.cat([global_vec_observations, obs['state'].unsqueeze(1)], dim=1)
    
    while global_step < args.total_timesteps:
        
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓     EVALUATION     ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓        
        
        if args.eval_freq > 0 and (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq:
            # evaluate
            rew_list = []
            
            actor.eval()
            stime = time.perf_counter()
            
            eval_obs, _ = eval_envs.reset()
            
            _h,_w,_c = envs.single_observation_space[args.obs_mode].shape
            eval_img_obs = torch.empty((args.num_eval_envs, 0, _h, _w, _c)).to(device)   #n_e, 0, s_d
            eval_img_obs = torch.cat([eval_img_obs, eval_obs[args.obs_mode].unsqueeze(1)], dim=1)
            
            eval_vec_obs = torch.empty((args.num_eval_envs, 0, eval_envs.single_observation_space['state'].shape[0])).to(device)   #n_e, 0, s_d
            eval_vec_obs = torch.cat([eval_vec_obs, eval_obs['state'].unsqueeze(1)], dim=1)
            
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):  #num_eval_steps = 50
                
                if eval_vec_obs.shape[1] > args.seq_len:
                    eval_img_obs = eval_img_obs[:, -args.seq_len:,]
                    eval_vec_obs = eval_vec_obs[:, -args.seq_len:,]
                
                
                with torch.no_grad():
                    obs4actor = {'state': eval_vec_obs, args.obs_mode: eval_img_obs}
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(actor.get_eval_action(obs4actor))
                    
                    eval_img_obs = torch.cat([eval_img_obs, eval_obs[args.obs_mode].unsqueeze(1)], dim=1)
                    eval_vec_obs = torch.cat([eval_vec_obs, eval_obs['state'].unsqueeze(1)], dim=1)
                    
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
            
            if global_vec_observations.shape[1] > args.seq_len:
                global_vec_observations = global_vec_observations[:, -args.seq_len-1:,]
                global_img_observations = global_img_observations[:, -args.seq_len-1:,]
            
            #print(f"local_step {local_step} out of {args.steps_per_env}")
            vec_obs2act = smart_slice(global_vec_observations, args.seq_len, info['elapsed_steps'].tolist(), for_rb = False)
            img_obs2act = smart_slice(global_img_observations, args.seq_len, info['elapsed_steps'].tolist(), for_rb = False)
            
            if not learning_has_started:
                actions = torch.tensor(envs.action_space.sample(), dtype=torch.float32, device=device)
            else:
                obs4act = {'state': vec_obs2act, args.obs_mode: img_obs2act}
                actions, _, _, _ = actor.get_action(obs4act)
                actions = actions.detach()

            
            
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)  #elapsed_steps = tensor([n, 0, s, 0, m])
            
            
            global_img_observations = torch.cat([global_img_observations, next_obs[args.obs_mode].unsqueeze(1)], dim=1)
            global_vec_observations = torch.cat([global_vec_observations, next_obs['state'].unsqueeze(1)], dim=1)
            
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
            
            '''
            real_next_obs содержат 51 степ
            next_obs не содержат 51 степ
            '''
                        
            real_img_global_observations = global_img_observations.clone()
            real_vec_global_observations = global_vec_observations.clone()
            
            real_img_global_observations[:,-1,:] = real_next_obs[args.obs_mode]
            real_vec_global_observations[:,-1,:] = real_next_obs['state']
            
            img_obs2RB, img_n_obs2RB = smart_slice(real_img_global_observations, args.seq_len, infos['elapsed_steps'].tolist(), for_rb=True)
            
            vec_obs2RB, vec_n_obs2RB = smart_slice(real_vec_global_observations, args.seq_len, infos['elapsed_steps'].tolist(), for_rb=True)
            
            obs2RB = {'state': vec_obs2RB, args.obs_mode: img_obs2RB}
            n_obs2RB = {'state': vec_n_obs2RB, args.obs_mode: img_n_obs2RB}
            
            
            rb.add(obs2RB, n_obs2RB, actions, rewards, stop_bootstrap)   # RB* <-- (ne,cont,sd), (ne,cont,sd), (ne,ad), (ne), (ne) 
            
            
            if 'episode' in infos and infos['episode']['success_once'].any():
                success_indices = torch.where(infos['episode']['success_once'])[0].tolist() # выводим индексы тех сред где случился success
                reseted_obs, infos = envs.reset(options={"env_idx":success_indices}) # обновляем только те среды, в который словили sr_once. при этом остальныве среды остаются неизменными
                global_img_observations[:,-1,:] = reseted_obs[args.obs_mode]
                global_vec_observations[:,-1,:] = reseted_obs['state']

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        pbar.update(args.num_envs * args.steps_per_env)


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