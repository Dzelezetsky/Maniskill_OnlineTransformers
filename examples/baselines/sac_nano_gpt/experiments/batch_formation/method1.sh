#!/bin/bash

seed=(1 2 3)

for seed in ${seed[@]}
do

    CUDA_VISIBLE_DEVICES=0 python examples/baselines/sac_nano_gpt/D_ver5_NEW_method1.py --env_id="PickCube-v1" --no-track \
        --total_timesteps=350_000  --num_envs=15 --num_eval_envs=100 --eval_freq=10_000 \
        --no-partial_reset --no-eval_partial_reset --control_mode="pd_ee_delta_pos" \
        --utd=0.5 --buffer_size=1_000_000 --num-steps=50 --num_eval_steps=50 \
        --seed $seed

  done