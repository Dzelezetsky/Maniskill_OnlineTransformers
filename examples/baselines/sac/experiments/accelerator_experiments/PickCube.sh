#!/bin/bash


# PushT-v1 StackCube-v1 PokeCube-v1 LiftPegUpright-v1 OpenCabinetDrawer-v1 TwoRobotPickCube-v1: PegInsertionSide-v1

CUDA_DEV=$1

seed=(3)

for seed in ${seed[@]}
do

    CUDA_VISIBLE_DEVICES=$CUDA_DEV python examples/baselines/sac/sac_rgbd.py --env_id="PickCube-v1" --obs_mode="rgb" \
        --num_envs=32 --utd=0.5 --buffer_size=300_000 \
        --control-mode="pd_ee_delta_pos" --camera_width=64 --camera_height=64 \
        --total_timesteps=1_000_000 --eval_freq=10_000 \
        --seed $seed

  done

