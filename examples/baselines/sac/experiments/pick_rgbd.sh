#!/bin/bash



CUDA_DEV=$1

seed=(3)

for seed in ${seed[@]}
do

    CUDA_VISIBLE_DEVICES=$CUDA_DEV python examples/baselines/sac/sac_rgbd.py --env_id="PickCube-v1" --obs_mode="rgb" --camera_width=64 --camera_height=64 \
        --total_timesteps=1_500_000 --buffer_size=900_000 --num_envs=384 --num_eval_envs=384 --training_freq=768 \
        --no-partial_reset --no-eval_partial_reset --control-mode="pd_ee_delta_pos" \
        --num-steps=50 --num_eval_steps=50 \
        --utd=0.5 --eval_freq=10_000 \
        --seed $seed\ 
        

  done