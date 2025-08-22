#!/bin/bash


# PushT-v1 StackCube-v1 PokeCube-v1 LiftPegUpright-v1 OpenCabinetDrawer-v1 TwoRobotPickCube-v1: PegInsertionSide-v1

CUDA_DEV=$1

seed=(2 3)

for seed in ${seed[@]}
do

    CUDA_VISIBLE_DEVICES=$CUDA_DEV CUDA_LAUNCH_BLOCKING=1 python acc_rgb.py --env_id="PullCube-v1" --obs_mode="rgb" \
        --num_envs=350 --num_eval_envs=350 --utd=0.5 --buffer_size=300_000 --training_freq=700 \
        --control-mode="pd_ee_delta_pos" --camera_width=64 --camera_height=64 \
        --total_timesteps=1_000_000 --eval_freq=10_000 \
        --seed $seed

  done

