#!/bin/bash


# PushT-v1 StackCube-v1 PokeCube-v1 LiftPegUpright-v1 OpenCabinetDrawer-v1 TwoRobotPickCube-v1: PegInsertionSide-v1

CUDA_DEV=$1

seed=(1 2 3)

for seed in ${seed[@]}
do

    CUDA_VISIBLE_DEVICES=$CUDA_DEV python acc_vec.py --env_id="PushCube-v1" --no-track --loss_proportion="bc" \
        --num_envs=300 --num_eval_envs=300 --training_freq=600 --utd=0.5 --buffer_size=200_000 \
        --no-partial_reset --no-eval_partial_reset --control-mode="pd_ee_delta_pos" \
        --total_timesteps=15_000_000 --eval_freq=10_000 --num-steps=50 --num_eval_steps=50 \
        --seed $seed

  done




 