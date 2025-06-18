#!/bin/bash


# PushT-v1 StackCube-v1 PokeCube-v1 LiftPegUpright-v1 OpenCabinetDrawer-v1 TwoRobotPickCube-v1: PegInsertionSide-v1
#--checkpoint='runs/[LAST_PAD]PickCube-v1__D_ver5_RGBD_NEW_last_pad__1__1748516084/ckpt_480000.pt'\

CUDA_DEV=$1

seed=(3)

for seed in ${seed[@]}
do

    CUDA_VISIBLE_DEVICES=$CUDA_DEV python examples/baselines/sac_nano_gpt/D_ver5_RGBD_NEW_last_pad.py --env_id="PokeCube-v1" --obs_mode="rgb" --camera_width=64 --camera_height=64 \
        --total_timesteps=5_000_000 --buffer_size=900_000 --num_envs=384 --num_eval_envs=384 --training_freq=768 \
        --no-partial_reset --no-eval_partial_reset --control-mode="pd_ee_delta_pos" \
        --num-steps=50 --num_eval_steps=50 \
        --utd=0.5 --eval_freq=10_000 \
        --seq_len=3 --seed $seed\ 
        

  done