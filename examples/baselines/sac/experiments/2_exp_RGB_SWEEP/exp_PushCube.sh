#!/bin/bash


# PushT-v1 StackCube-v1 PokeCube-v1 LiftPegUpright-v1 OpenCabinetDrawer-v1 TwoRobotPickCube-v1: PegInsertionSide-v1



seed=(1 2 3)

for seed in ${seed[@]}
do

    # python ../../sac_rgbd.py --env_id="PushCube-v1" --obs_mode="rgb" --camera_width=64 --camera_height=64 \
    #     --total_timesteps=2_000_000 --buffer_size=300_000 --num_envs=32 --num_eval_envs=100 \
    #     --no-partial_reset --no-eval_partial_reset --no-save_model --control-mode="pd_ee_delta_pos" \
    #     --num-steps=50 --num_eval_steps=50 \
    #     --utd=0.5 --eval_freq=10_000 \
    #     --seed $seed
    python ../../sac_rgbd.py --env_id="PushCube-v1" --obs_mode="rgb" --camera_width=64 --camera_height=64 \
        --total_timesteps=2_000_000 --buffer_size=50_000 --num_envs=4 --num_eval_envs=4 \
        --no-partial_reset --no-eval_partial_reset --no-save_model --control-mode="pd_ee_delta_pos" \
        --num-steps=50 --num_eval_steps=50 \
        --utd=0.5 --eval_freq=10_000 \
        --seed $seed

  done
