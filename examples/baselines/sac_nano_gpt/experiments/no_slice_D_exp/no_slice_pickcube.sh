#!/bin/bash



seed=(1 2 3)

for seed in ${seed[@]}
do

    CUDA_VISIBLE_DEVICES=1 python examples/baselines/sac_nano_gpt/D_ver5_no_slice_at_all.py --env_id="PickCube-v1" --no-track \
        --total_timesteps=1_000_000  --num_envs=32 --num_eval_envs=100 --eval_freq=10_000 \
        --no-partial_reset --no-eval_partial_reset --control_mode="pd_ee_delta_pos" \
        --utd=0.5 --buffer_size=1_000_000 --num-steps=50 --num_eval_steps=50 \
        --seq_len 10 --n_layer 1 --n_head 2 --n_embd 256 --dropout 0.0 --bias \
        --seed $seed

  done