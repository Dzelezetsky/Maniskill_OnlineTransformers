#!/bin/bash


# PushT-v1 StackCube-v1 PokeCube-v1 LiftPegUpright-v1 OpenCabinetDrawer-v1 TwoRobotPickCube-v1: PegInsertionSide-v1



seed=(1 2 3)

for seed in ${seed[@]}
do

    CUDA_VISIBLE_DEVICES=1 python examples/baselines/sac_nano_gpt/D_lstm_ver5.py --env_id="TriFingerRotateCubeLevel0-v1" --no-track \
        --total_timesteps=5_000_000  --num_envs=32 --num_eval_envs=100 --eval_freq=10_000 \
        --no-partial_reset --no-eval_partial_reset --control_mode="pd_joint_delta_pos" \
        --utd=0.5 --buffer_size=1_000_000 --num-steps=250 --num_eval_steps=250 \
        --seed $seed

done