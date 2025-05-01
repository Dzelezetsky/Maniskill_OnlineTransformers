python examples/baselines/sac_nano_gpt/D_ver5_NEW_last_pad.py --env_id="PickCube-v1" --no-track --total_timesteps=350_000  --num_envs=15 \
    --num_eval_envs=100 --eval_freq=10_000 --no-partial_reset --no-eval_partial_reset --control_mode="pd_ee_delta_pos" --utd=0.5 --buffer_size=1_000_000 \
    --num-steps=50 --num_eval_steps=50 --seed=1 --evaluate \
    --checkpoint="/home/mount/Maniskill_OnlineTransformers/runs/[NO_SLICE_AT_ALL]PickCube-v1__D_ver5_NEW_clear_pad__3__1746046809/final_ckpt.pt"
    #--checkpoint="/home/mount/Maniskill_OnlineTransformers/runs/[LAST_PAD]PickCube-v1__D_ver5_NEW_last_pad__3__1746066318/final_ckpt.pt"
    #--checkpoint="/home/mount/Maniskill_OnlineTransformers/runs/[ZERO_PAD]PickCube-v1__D_ver5_NEW_zero_pad__3__1746061755/final_ckpt.pt"