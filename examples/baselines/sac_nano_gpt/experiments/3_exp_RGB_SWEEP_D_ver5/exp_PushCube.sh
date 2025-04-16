#!/bin/bash


# PushT-v1 StackCube-v1 PokeCube-v1 LiftPegUpright-v1 OpenCabinetDrawer-v1 TwoRobotPickCube-v1: PegInsertionSide-v1



seed=(1 2 3)

for seed in ${seed[@]}
do

    # python ../../D_ver5_rgbd.py --env_id="PushCube-v1" --obs_mode="rgb" --camera_width=64 --camera_height=64 \
    #     --total_timesteps=2_000_000 --buffer_size=300_000 --num_envs=32 --num_eval_envs=100 \
    #     --no-partial_reset --no-eval_partial_reset --no-save_model --control-mode="pd_ee_delta_pos" \
    #     --seq_len 10 --n_layer 1 --n_head 2 --n_embd 256 --dropout 0.0 --bias \
    #     --num-steps=50 --num_eval_steps=50 \
    #     --utd=0.5 --eval_freq=10_000 \
    #     --seed $seed

    python ../../D_ver5_rgbd.py --env_id="PushCube-v1" --obs_mode="rgb" --camera_width=64 --camera_height=64 \
        --total_timesteps=2_000_000 --buffer_size=100_000 --num_envs=32 --num_eval_envs=100 \
        --no-partial_reset --no-eval_partial_reset --no-save_model --control-mode="pd_ee_delta_pos" \
        --seq_len 10 --n_layer 1 --n_head 2 --n_embd 256 --dropout 0.0 --bias \
        --num-steps=50 --num_eval_steps=50 \
        --utd=0.5 --eval_freq=10_000 \
        --seed $seed \
        --no-include-state

  done
