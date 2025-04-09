#!/bin/bash


# PushT-v1 StackCube-v1 PokeCube-v1 LiftPegUpright-v1 OpenCabinetDrawer-v1 TwoRobotPickCube-v1: PegInsertionSide-v1



seed=(1 2 3)

for seed in ${seed[@]}
do


    python examples/baselines/sac_nano_gpt/D_ver5.py --env_id="PokeCube-v1" --no-track \
        --total_timesteps=5_000_000  --num_envs=32 --num_eval_envs=100 --eval_freq=10_000 \
        --no-partial_reset --no-eval_partial_reset --control_mode="pd_ee_delta_pos" \
        --utd=0.5 --buffer_size=1_000_000 --num-steps=20 --num_eval_steps=50 \
        --seq_len 10 --n_layer 1 --n_head 2 --n_embd 256 --dropout 0.0 --bias \
        --seed $seed

  done
