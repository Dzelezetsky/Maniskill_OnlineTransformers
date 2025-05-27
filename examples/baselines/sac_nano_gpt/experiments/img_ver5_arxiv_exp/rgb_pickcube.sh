#!/bin/bash


# PushT-v1 StackCube-v1 PokeCube-v1 LiftPegUpright-v1 OpenCabinetDrawer-v1 TwoRobotPickCube-v1: PegInsertionSide-v1



seed=(1)

for seed in ${seed[@]}
do

    CUDA_VISIBLE_DEVICES=0 python examples/baselines/sac_nano_gpt/D_ver5_RGBD_NEW_last_pad.py --env_id="PickCube-v1" --obs_mode="rgb" --camera_width=64 --camera_height=64 \
        --total_timesteps=2_000_000 --buffer_size=200_000 --num_envs=30 --num_eval_envs=30 \
        --no-partial_reset --no-eval_partial_reset --control-mode="pd_ee_delta_pos" \
        --num-steps=50 --num_eval_steps=50 \
        --utd=0.5 --eval_freq=10_000 \
        --seed $seed

  done