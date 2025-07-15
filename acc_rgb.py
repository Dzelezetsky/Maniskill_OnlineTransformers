

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
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import time
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
    env_vectorization: str = "gpu"
    """the type of environment vectorization to use"""
    num_envs: int = 16
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
    eval_freq: int = 25
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
# @dataclass
# class ReplayBufferSample:
#     obs: torch.Tensor
#     next_obs: torch.Tensor
#     actions: torch.Tensor
#     rewards: torch.Tensor
#     dones: torch.Tensor
# class ReplayBuffer:
#     def __init__(self, env, num_envs: int, buffer_size: int, storage_device: torch.device, sample_device: torch.device):
#         self.buffer_size = buffer_size
#         self.pos = 0
#         self.full = False
#         self.num_envs = num_envs
#         self.storage_device = storage_device
#         self.sample_device = sample_device
#         self.per_env_buffer_size = buffer_size // num_envs
#         # note 128x128x3 RGB data with replay buffer size 100_000 takes up around 4.7GB of GPU memory
#         # 32 parallel envs with rendering uses up around 2.2GB of GPU memory.
#         self.obs = DictArray((self.per_env_buffer_size, num_envs), env.single_observation_space, device=storage_device)
#         # TODO (stao): optimize final observation storage
#         self.next_obs = DictArray((self.per_env_buffer_size, num_envs), env.single_observation_space, device=storage_device)
#         self.actions = torch.zeros((self.per_env_buffer_size, num_envs) + env.single_action_space.shape, device=storage_device)
#         self.logprobs = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
#         self.rewards = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
#         self.dones = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
#         self.values = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)

#     def add(self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor):
#         if self.storage_device == torch.device("cpu"):
#             obs = {k: v.cpu() for k, v in obs.items()}
#             next_obs = {k: v.cpu() for k, v in next_obs.items()}
#             action = action.cpu()
#             reward = reward.cpu()
#             done = done.cpu()

#         self.obs[self.pos] = obs
#         self.next_obs[self.pos] = next_obs

#         self.actions[self.pos] = action
#         self.rewards[self.pos] = reward
#         self.dones[self.pos] = done

#         self.pos += 1
#         if self.pos == self.per_env_buffer_size:
#             self.full = True
#             self.pos = 0
#     def sample(self, batch_size: int):
#         if self.full:
#             batch_inds = torch.randint(0, self.per_env_buffer_size, size=(batch_size, ))
#         else:
#             batch_inds = torch.randint(0, self.pos, size=(batch_size, ))
#         env_inds = torch.randint(0, self.num_envs, size=(batch_size, ))
#         obs_sample = self.obs[batch_inds, env_inds]
#         next_obs_sample = self.next_obs[batch_inds, env_inds]
#         obs_sample = {k: v.to(self.sample_device) for k, v in obs_sample.items()}
#         next_obs_sample = {k: v.to(self.sample_device) for k, v in next_obs_sample.items()}
#         return ReplayBufferSample(
#             obs=obs_sample,
#             next_obs=next_obs_sample,
#             actions=self.actions[batch_inds, env_inds].to(self.sample_device),
#             rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
#             dones=self.dones[batch_inds, env_inds].to(self.sample_device)
#         )

@dataclass
class ReplayBufferSample:
    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
class ReplayBuffer:
    def __init__(self, env, num_envs: int, buffer_size: int, storage_device: torch.device, sample_device: torch.device):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.env = env
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device
        self.per_env_buffer_size = buffer_size // num_envs
        # note 128x128x3 RGB data with replay buffer size 100_000 takes up around 4.7GB of GPU memory
        # 32 parallel envs with rendering uses up around 2.2GB of GPU memory.
        self.obs = DictArray((self.per_env_buffer_size, num_envs), env.single_observation_space, device=storage_device)
        # TODO (stao): optimize final observation storage
        self.next_obs = DictArray((self.per_env_buffer_size, num_envs), env.single_observation_space, device=storage_device)
        self.actions = torch.zeros((self.per_env_buffer_size, num_envs) + env.single_action_space.shape, device=storage_device)
        self.logprobs = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        self.rewards = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        self.dones = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        self.values = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        
        self.associated_r = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        self.Q = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        self.V = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)


    def add_associated_reward(self, positions: list[int], rewards: torch.Tensor):
        """
        Добавляет к associated_r для каждого шага в positions соответствующее `rewards`.
        Умеет работать с любым порядком позиций, в том числе с переходом через конец буфера.
        """
        if len(positions) == 0:
            return

        # приводим к тензору нужного устройства
        pos = torch.tensor(positions, dtype=torch.long, device=self.storage_device)  # [T]
        # rewards: [N] -> [T, N]
        R = rewards.unsqueeze(0).expand(len(positions), -1)

        # аккуратно добавляем
        # self.associated_r.shape == [buffer_len, num_envs]
        self.associated_r[pos, :] += R
        
    def add_qv_estimates(self, positions: list[int]):
        """
        Использует associated_r и rewards для вычисления Q(s,a) и V(s) для каждого шага в positions.
        Позиции могут «обходить» конец буфера; мы сначала перестроим их в хронологическом порядке,
        потом посчитаем кумулятивные суммы и запишем Q/V обратно в оригинальные слоты.
        """
        if not positions:
            return

        # Находим точку «склейки» — где идёт спад индексов
        split_idx = None
        for i in range(len(positions) - 1):
            if positions[i] > positions[i + 1]:
                split_idx = i + 1
                break

        if split_idx is None:
            ordered = positions[:]            # уже упорядочено
        else:
            # переставим хвост вперёд
            ordered = positions[split_idx:] + positions[:split_idx]

        # тензоры для индексов
        pos = torch.tensor(ordered, dtype=torch.long, device=self.storage_device)  # [T]
        envs = torch.arange(self.num_envs, device=self.storage_device)            # [N]
        ti, ei = torch.meshgrid(pos, envs, indexing="ij")                         # [T, N]

        # достаём соответствующие матрицы
        R = self.rewards[ti, ei]         # [T, N]
        A = self.associated_r[ti, ei]    # [T, N]

        # Q = A − cumsum(R) без текущего, V = A − cumsum(R) с текущим
        cum_excl = torch.cumsum(R, dim=0) - R   # [T, N]
        cum_incl = torch.cumsum(R, dim=0)       # [T, N]

        Q_vals = A - cum_excl
        V_vals = A - cum_incl

        # Записываем обратно в буфер, сопоставляя в том же порядке `ordered`
        # self.Q.shape == [buffer_len, num_envs]
        self.Q[pos, :] = Q_vals
        self.V[pos, :] = V_vals
    
        
    
    def add(self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor):
        if self.storage_device == torch.device("cpu"):
            obs = {k: v.cpu() for k, v in obs.items()}
            next_obs = {k: v.cpu() for k, v in next_obs.items()}
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()
        if self.pos+args.num_steps >= self.per_env_buffer_size:
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
            return self.per_env_buffer_size 
        
        return self.pos-1
    
        
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
      
      
      
    def make_sequential_dataloader(self,
                               positions: list[int],
                               context_len: int,
                               batch_size: int,
                               shuffle: bool = True):
        if not positions:
            raise ValueError("Positions list is empty.")

        # 1) Найти, есть ли «склейка» (wrap-around) в списке позиций
        split_idx = None
        for i in range(len(positions) - 1):
            if positions[i] > positions[i + 1]:
                split_idx = i + 1
                break

        # 2) Перестроить список в хронологическом порядке
        if split_idx is not None:
            ordered = positions[split_idx:] + positions[:split_idx]
        else:
            ordered = positions[:]

        # 3) Собираем индексы по времени и окружениям
        pos = torch.tensor(ordered, dtype=torch.long, device=self.storage_device)  # [T]
        num_steps = len(pos)
        n_envs = self.num_envs
        device = self.sample_device

        time_idx, env_idx = torch.meshgrid(
            pos,
            torch.arange(n_envs, device=self.storage_device),
            indexing="ij"
        )  # оба [T, N]

        # 4) Выдёргиваем всё из кольцевого буфера
        obs          = self.obs[time_idx, env_idx]
        next_obs     = self.next_obs[time_idx, env_idx]
        actions      = self.actions[time_idx, env_idx]
        rewards      = self.rewards[time_idx, env_idx]
        dones        = self.dones[time_idx, env_idx]
        associated_r = self.associated_r[time_idx, env_idx]
        q_vals       = self.Q[time_idx, env_idx]
        v_vals       = self.V[time_idx, env_idx]

        # 5) Формируем скользящие окна длины context_len
        sequences = []
        for start in range(num_steps - context_len + 1):
            end = start + context_len

            # из [T, N, ...] → [N, T, ...]
            obs_seq        = {k: v[start:end].to(device).permute(1, 0, *range(2, v.ndim))
                            for k, v in obs.items()}
            next_obs_seq   = {k: v[start:end].to(device).permute(1, 0, *range(2, v.ndim))
                            for k, v in next_obs.items()}
            actions_seq    = actions[start:end].to(device).permute(1, 0, *range(2, actions.ndim))
            rewards_seq    = rewards[start:end].to(device).permute(1, 0)
            dones_seq      = dones[start:end].to(device).permute(1, 0)
            assoc_r_seq    = associated_r[start:end].to(device).permute(1, 0)
            q_seq          = q_vals[start:end].to(device).permute(1, 0)
            v_seq          = v_vals[start:end].to(device).permute(1, 0)

            # разбить по env и собрать список последовательностей
            for env in range(n_envs):
                sequences.append((
                    {k: obs_seq[k][env]       for k in obs_seq},
                    {k: next_obs_seq[k][env]  for k in next_obs_seq},
                    actions_seq[env],
                    rewards_seq[env],
                    dones_seq[env],
                    assoc_r_seq[env],
                    q_seq[env],
                    v_seq[env]
                ))

        # 6) Переводим всё в батч
        obs_batch        = {k: torch.stack([s[0][k] for s in sequences]) for k in sequences[0][0]}
        next_obs_batch   = {k: torch.stack([s[1][k] for s in sequences]) for k in sequences[0][1]}
        actions_batch    = torch.stack([s[2] for s in sequences])
        rewards_batch    = torch.stack([s[3] for s in sequences])
        dones_batch      = torch.stack([s[4] for s in sequences])
        assoc_r_batch    = torch.stack([s[5] for s in sequences])
        q_batch          = torch.stack([s[6] for s in sequences])
        v_batch          = torch.stack([s[7] for s in sequences])

        dataset = TensorDataset(
            *list(obs_batch.values()),
            *list(next_obs_batch.values()),
            actions_batch,
            rewards_batch,
            dones_batch,
            assoc_r_batch,
            q_batch,
            v_batch
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


    
    # def sample_for_trans(self, positions: list):    
    #     positions = torch.tensor(positions)  # T
    #     env_ids = torch.arange(self.num_envs)  # N
    #     ti, ei = torch.meshgrid(positions, env_ids, indexing="ij")  # T, N

    #     obs_batch = self.obs[ti, ei]
    #     next_obs_batch = self.next_obs[ti, ei]
    #     actions_batch = self.actions[ti, ei]
    #     rewards_batch = self.rewards[ti, ei]
    #     dones_batch = self.dones[ti, ei]

    #     obs_batch = {k: v.to(self.sample_device) for k, v in obs_batch.items()}
    #     next_obs_batch = {k: v.to(self.sample_device) for k, v in next_obs_batch.items()}

    #     return ReplayBufferSample(
    #         obs=obs_batch,
    #         next_obs=next_obs_batch,
    #         actions=actions_batch.to(self.sample_device),
    #         rewards=rewards_batch.to(self.sample_device),
    #         dones=dones_batch.to(self.sample_device),
    #     )
        
    # def make_sequential_dataloader(self, positions: list[int], context_len: int, batch_size: int, shuffle: bool = True):
    #     pos = torch.tensor(positions)
    #     num_steps = len(pos)
    #     n_envs = self.num_envs
    #     device = self.sample_device   
        
    #     time_idx, env_idx = torch.meshgrid(pos, torch.arange(n_envs), indexing="ij")
        
    #     obs = self.obs[time_idx, env_idx]
    #     next_obs = self.next_obs[time_idx, env_idx]
    #     actions = self.actions[time_idx, env_idx]
    #     rewards = self.rewards[time_idx, env_idx]
    #     dones = self.dones[time_idx, env_idx]
    #     associated_r = self.associated_r[time_idx, env_idx]
    #     q_vals = self.Q[time_idx, env_idx]
    #     v_vals = self.V[time_idx, env_idx]

    #     sequences = []
    #     for start in range(num_steps - context_len + 1):
    #         end = start + context_len

    #         # Собираем последовательности: [context_len, n_envs, ...] → [n_envs, context_len, ...]
    #         obs_seq = {k: v[start:end].to(device).permute(1, 0, *range(2, v.ndim)) for k, v in obs.items()}
    #         next_obs_seq = {k: v[start:end].to(device).permute(1, 0, *range(2, v.ndim)) for k, v in next_obs.items()}
    #         actions_seq = actions[start:end].to(device).permute(1, 0, *range(2, actions.ndim))
    #         rewards_seq = rewards[start:end].to(device).permute(1, 0)
    #         dones_seq = dones[start:end].to(device).permute(1, 0)
    #         associated_r_seq = associated_r[start:end].to(device).permute(1, 0)
    #         q_seq = q_vals[start:end].to(device).permute(1, 0)
    #         v_seq = v_vals[start:end].to(device).permute(1, 0)

    #         for i in range(n_envs):
    #             seq = (
    #                 {k: v[i] for k, v in obs_seq.items()},           # [context, ...]
    #                 {k: v[i] for k, v in next_obs_seq.items()},
    #                 actions_seq[i],
    #                 rewards_seq[i],
    #                 dones_seq[i],
    #                 associated_r_seq[i],
    #                 q_seq[i],
    #                 v_seq[i]
    #             )
    #             sequences.append(seq)

    #     obs_seq = {k: torch.stack([s[0][k] for s in sequences]) for k in sequences[0][0]}
    #     next_obs_seq = {k: torch.stack([s[1][k] for s in sequences]) for k in sequences[0][1]}
    #     actions_seq = torch.stack([s[2] for s in sequences])
    #     rewards_seq = torch.stack([s[3] for s in sequences])
    #     dones_seq = torch.stack([s[4] for s in sequences])
    #     associated_r_seq = torch.stack([s[5] for s in sequences]) 
    #     q_seq = torch.stack([s[6] for s in sequences])
    #     v_seq = torch.stack([s[7] for s in sequences])   

    #     dataset = TensorDataset(
    #         *list(obs_seq.values()),
    #         *list(next_obs_seq.values()),
    #         actions_seq,
    #         rewards_seq,
    #         dones_seq,
    #         associated_r_seq,
    #         q_seq,
    #         v_seq
    #     )
    #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    #     return dataloader
    
    
    
########## SPECIAL TOOLS FOR TRANSFORMER BELOW  ##################

class Trans_PlainConv(nn.Module):
    '''
    Conv Net constructor
    '''
    def __init__(self,
                 in_channels=3,
                 out_dim=228, #256,
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

class Trans_EncoderObsWrapper(nn.Module):
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
        
        encoder = Trans_EncoderObsWrapper
        
        
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
        
        #print(f"x: {x.shape}") 
        #print(f"pe: {pos_emb.shape}")  
        
        x = self.drop(x + pos_emb)
        for block in self.transformer_layers:
            x = block(x)
        x = self.ln_f(x)

        return x
    
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

        return torch.mul(1 - z, x) + torch.mul(z, h)
        
    
class Trans_Actor(nn.Module):
    def __init__(self, envs, args, sample_obs):
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

        self.encoder = Trans_EncoderObsWrapper(
            Trans_PlainConv(in_channels=in_channels, out_dim=228, image_size=image_size) # assume image is 64x64
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
        self.action_scale = self.action_scale.to(mean.device)
        self.action_bias = self.action_bias.to(mean.device)
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
    


class Trans_SoftQNetwork(nn.Module):
    '''
    Q-network for Transformer-based maniskill tasks
    '''
    def __init__(self, env, args, encoder: Trans_EncoderObsWrapper):
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

    
##################### TOOLS FOR TRANS  ABOVE ################################## 


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
        
        
        #print(x.shape)
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
class EncoderObsWrapper(nn.Module):
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
        img = img.permute(0, 3, 1, 2) # (B, C, H, W)
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

class SoftQNetwork(nn.Module):
    def __init__(self, envs, encoder: EncoderObsWrapper):
        super().__init__()
        self.encoder = encoder
        action_dim = np.prod(envs.single_action_space.shape)
        state_dim = envs.single_observation_space['state'].shape[0]
        self.mlp = make_mlp(encoder.encoder.out_dim+action_dim+state_dim, [512, 256, 1], last_act=False)

    def forward(self, obs, action, visual_feature=None, detach_encoder=False):
        if visual_feature is None:
            visual_feature = self.encoder(obs)
        if detach_encoder:
            visual_feature = visual_feature.detach()    
        x = torch.cat([visual_feature, obs["state"], action], dim=1)
        return self.mlp(x)


LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, envs, sample_obs):
        super().__init__()
        action_dim = np.prod(envs.single_action_space.shape)
        state_dim = envs.single_observation_space['state'].shape[0]
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
        self.mlp = make_mlp(self.encoder.encoder.out_dim+state_dim, [512, 256], last_act=True)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        # action rescaling
        self.action_scale = torch.FloatTensor((envs.single_action_space.high - envs.single_action_space.low) / 2.0)
        self.action_bias = torch.FloatTensor((envs.single_action_space.high + envs.single_action_space.low) / 2.0)

    def get_feature(self, obs, detach_encoder=False):
        visual_feature = self.encoder(obs)
        if detach_encoder:
            visual_feature = visual_feature.detach()
        x = torch.cat([visual_feature, obs['state']], dim=1)
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


def train_transformer(loss_proportion: int, positions: list, ascent_on: str, batch_size: int, context: int, shuffle: bool):
    
    '''
    obs_s     torch.Size([bs, cont, 29]) 
    obs_i     torch.Size([bs, cont, 64, 64, 3]) 
    n_obs_s   torch.Size([bs, cont, 29]) 
    n_obs_i   torch.Size([bs, cont, 64, 64, 3]) 
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
        obs_s = batch[0] # state_based obs
        obs_i = batch[1] # image_based obs
        n_obs_s = batch[2] # state_based next obs
        n_obs_i = batch[3] # image_based next obs
        acts = batch[4]
        rew = batch[5]
        dones = batch[6]
        R = batch[7]  # Associated rewards with the episode under the positions
        Q = batch[8]  # Real Q values associated with states and observation on the positions
        V = batch[9]  # Real V values
    
        #ACTOR BC LOSS
        predicted_action, _, _, _ = trans_actor.get_action({'state': obs_s, 'rgb': obs_i})
        target_action = acts[:, -1].clone()  
        criterion = nn.MSELoss()
        
        #print(f'a bc {predicted_action.shape, target_action.shape}')
        trans_actor_BC_loss = criterion(predicted_action, target_action)
        
        trans_actor_optimizer.zero_grad()
        trans_actor_BC_loss.backward()
        trans_actor_optimizer.step()
        
        
        # CRITIC BC LOSS
        action_target = acts[:, -1].clone()

        q_predicted1 = trans_qf1({'state': obs_s, 'rgb': obs_i}, action_target)
        q_predicted2 = trans_qf2({'state': obs_s, 'rgb': obs_i}, action_target)

        # Only last timestep
        with torch.no_grad():
            q_target1 = qf1({'state': obs_s[:, -1].clone(), 'rgb': obs_i[:, -1].clone()}, action_target)
            q_target2 = qf2({'state': obs_s[:, -1].clone(), 'rgb': obs_i[:, -1].clone()}, action_target)

        #print(f'c bc {q_predicted1.shape, q_predicted1.shape}')
        trans_qf1_loss = F.mse_loss(q_predicted1, q_target1)
        trans_qf2_loss = F.mse_loss(q_predicted2, q_target2)
        trans_critic_BC_loss = trans_qf1_loss + trans_qf2_loss
        
        trans_q_optimizer.zero_grad()
        trans_critic_BC_loss.backward()
        trans_q_optimizer.step()
        
        
        #ACTOR RL LOSS
        pi, log_pi, _, visual_feature = trans_actor.get_action({'state':obs_s,'rgb':obs_i})
        
        pi_detached = pi.clone()
        if ascent_on == 'transformer':
            qf1_pi = trans_qf1({'state':obs_s,'rgb':obs_i}, pi_detached, visual_feature)  ##  detach_encoder=True ???????
            qf2_pi = trans_qf2({'state':obs_s,'rgb':obs_i}, pi_detached, visual_feature)##  detach_encoder=True ???????
        elif ascent_on == 'accelerator':
            print(obs_s.shape, obs_i.shape)
            qf1_pi = qf1({'state':obs_s[:,-1,],'rgb':obs_i[:,-1,]}, pi_detached, visual_feature)##  detach_encoder=True ???????
            qf2_pi = qf2({'state':obs_s[:,-1,],'rgb':obs_i[:,-1,]}, pi_detached, visual_feature)##  detach_encoder=True ???????
        
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
        Q_target = Q[:, -1].clone()
        action_target = acts[:, -1].clone()

        qf1_a_values = trans_qf1({'state': obs_s, 'rgb': obs_i}, action_target).reshape(-1)
        qf2_a_values = trans_qf2({'state': obs_s, 'rgb': obs_i}, action_target).reshape(-1)

        #print(f'c rl {qf1_a_values.shape, Q_target.shape}')
        qf1_loss = F.mse_loss(qf1_a_values, Q_target)
        qf2_loss = F.mse_loss(qf2_a_values, Q_target)
        trans_critic_RL_loss = qf1_loss + qf2_loss
        
        trans_q_optimizer.zero_grad()
        trans_critic_RL_loss.backward()
        trans_q_optimizer.step()
        
        
    duration = time.time() - start_time
    print(f"Transformer training completed in {duration:.2f} seconds.")
    logger.add_scalar("Trans/Actor_BC_loss", trans_actor_BC_loss.item(), global_step)
    logger.add_scalar("Trans/Critic_BC_loss", trans_critic_BC_loss.item(), global_step) 
    logger.add_scalar("Trans/Critic_RL_loss", trans_critic_RL_loss.item(), global_step)
    logger.add_scalar("Trans/Actor_RL_loss", trans_actor_RL_loss.item(), global_step)
        
        
            
def evaluate_transformer():
    # evaluate
    rew_list = []
    
    trans_actor.eval()
    stime = time.perf_counter()
    
    eval_obs, _ = trans_eval_envs.reset()
    
    _h,_w,_c = trans_eval_envs.single_observation_space[args.obs_mode].shape
    eval_img_obs = torch.empty((args.num_eval_envs, 0, _h, _w, _c)).to(device)   #n_e, 0, s_d
    eval_img_obs = torch.cat([eval_img_obs, eval_obs[args.obs_mode].unsqueeze(1)], dim=1)
    
    eval_vec_obs = torch.empty((args.num_eval_envs, 0, trans_eval_envs.single_observation_space['state'].shape[0])).to(device)   #n_e, 0, s_d
    eval_vec_obs = torch.cat([eval_vec_obs, eval_obs['state'].unsqueeze(1)], dim=1)
    
    eval_metrics = defaultdict(list)
    num_episodes = 0
    for _ in range(args.num_eval_steps):  #num_eval_steps = 50
        
        if eval_vec_obs.shape[1] > args.seq_len:
            eval_img_obs = eval_img_obs[:, -args.seq_len:,]
            eval_vec_obs = eval_vec_obs[:, -args.seq_len:,]
        
        
        with torch.no_grad():
            obs4actor = {'state': eval_vec_obs, args.obs_mode: eval_img_obs}
            eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = trans_eval_envs.step(trans_actor.get_eval_action(obs4actor))
            
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
        run_name = f"REAL_ACCELERATOR/RL_BC_{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    trans_eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, human_render_camera_configs=dict(shader_pack="default"), **env_kwargs)

    # rgbd obs mode returns a dict of data, we flatten it so there is just a rgbd key and state key
    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=args.include_state)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=args.include_state)
    trans_eval_envs = FlattenRGBDObservationWrapper(trans_eval_envs, rgb=True, depth=False, state=args.include_state)

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
        trans_eval_envs = RecordEpisode(trans_eval_envs, output_dir=eval_output_dir, save_trajectory=args.save_trajectory, save_video=args.capture_video, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
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

    # architecture is all actor, q-networks share the same vision encoder. Output of encoder is concatenates with any state data followed by separate MLPs.
    actor = Actor(envs, sample_obs=obs).to(device)
    qf1 = SoftQNetwork(envs, actor.encoder).to(device)
    qf2 = SoftQNetwork(envs, actor.encoder).to(device)
    qf1_target = SoftQNetwork(envs, actor.encoder).to(device)
    qf2_target = SoftQNetwork(envs, actor.encoder).to(device)
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
        
        
    # TRANSFORMER INITIALIZATION BELOW
    trans_actor = Trans_Actor(envs=eval_envs, args=args, sample_obs=obs).to(device)
    trans_qf1 = Trans_SoftQNetwork(eval_envs, args, trans_actor.encoder).to(device)
    trans_qf2 = Trans_SoftQNetwork(eval_envs, args, trans_actor.encoder).to(device)
        
    trans_qf1_target = Trans_SoftQNetwork(eval_envs, args, trans_actor.encoder).to(device)
    trans_qf2_target = Trans_SoftQNetwork(eval_envs, args, trans_actor.encoder).to(device)
        
    trans_q_optimizer = optim.Adam(
                        list(trans_qf1.transformer.parameters()) +
                        list(trans_qf2.transformer.parameters()) +
                        list(trans_qf1.net.parameters()) +
                        list(trans_qf2.net.parameters()) +
                        list(trans_qf1.encoder.parameters()),
                        lr=3e-4)
    trans_actor_optimizer = optim.Adam(list(trans_actor.parameters()), lr=3e-4)
        
    # # Automatic entropy tuning
    # if args.autotune:
    #     target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
    #     log_alpha = torch.zeros(1, requires_grad=True, device=device)
    #     alpha = log_alpha.exp().item()
    #     trans_a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    # else:
    #     alpha = args.alpha

    # TRANSFORMER INITIALIZATION ABOVE    

    global_step = 0
    global_update = 0
    learning_has_started = False

    global_steps_per_iteration = args.num_envs * (args.steps_per_env)
    pbar = tqdm.tqdm(range(args.total_timesteps))
    cumulative_times = defaultdict(float)

    while global_step < args.total_timesteps:
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
                    # Добавляем в буфер (как в первом варианте)
                    real_next_obs = {k: v.clone() for k, v in eval_next_obs.items()}

                    # По твоей логике bootstrap_at_done = "always"
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
            
            print(positions)
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
            
            train_transformer(loss_proportion=1, positions=positions, ascent_on='transformer', batch_size=100, context=3, shuffle=True)
            evaluate_transformer()
            

            # if args.save_model:
            #     model_path = f"runs/{run_name}/ckpt_{global_step}.pt"
            #     torch.save({
            #         'actor': actor.state_dict(),
            #         'qf1': qf1_target.state_dict(),
            #         'qf2': qf2_target.state_dict(),
            #         'log_alpha': log_alpha,
            #     }, model_path)
            #     print(f"model saved to {model_path}")

        # Collect samples from environemnts
        rollout_time = time.perf_counter()
        for local_step in range(args.steps_per_env):
            global_step += 1 * args.num_envs

            # ALGO LOGIC: put action logic here
            if not learning_has_started:
                actions = torch.tensor(envs.action_space.sample(), dtype=torch.float32, device=device)
            else:
                actions, _, _, _ = actor.get_action(obs)
                actions = actions.detach()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
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

            rb.add(obs, real_next_obs, actions, rewards, stop_bootstrap)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        pbar.update(args.num_envs * args.steps_per_env)

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