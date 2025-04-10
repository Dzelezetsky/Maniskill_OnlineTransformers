#!/bin/bash

echo $(nvidia-smi)

echo $(cat /usr/share/vulkan/icd.d/nvidia_icd.json)

# Set environment variable to skip prompts
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1

# Get home directory
HOME_DIR="$HOME"

# echo "Downloading assets..."
echo "HOME_DIR: $HOME_DIR"

# Initialize conda for the current shell session
eval "$(conda shell.bash hook)"

# Install huggingface-cli if not already installed
if ! command -v huggingface-cli &> /dev/null; then
    pip install huggingface_hub
fi

# Create directory
mkdir -p "$HOME_DIR/.maniskill/data/assets"

# Download assets using Python API instead of CLI
python3 -c "
from huggingface_hub import hf_hub_download
import os

output_file = hf_hub_download(
    repo_id='haosulab/ManiSkill2',
    filename='data/mani_skill2_ycb.zip',
    repo_type='dataset',
    local_dir='/tmp/maniskill_download'
)
print(f'Downloaded to: {output_file}')
"

# Copy the downloaded file
cp /tmp/maniskill_download/data/mani_skill2_ycb.zip "$HOME_DIR/.maniskill/data/assets/"

# Extract the zip file
unzip "$HOME_DIR/.maniskill/data/assets/mani_skill2_ycb.zip" -d "$HOME_DIR/.maniskill/data/assets/"

# Clean up temporary files
rm -rf /tmp/maniskill_download

# Initialize and activate conda environment
conda init bash
source ~/.bashrc
conda activate mikasa

# pip install -e .

pip install mikasa_robo_suite
pip install gymnasium==0.29.1

# # Run the main script
# python3 mikasa_robo_suite/dataset_collectors/parallel_dataset_collection_manager.py \
#     --path-to-save-data="/workspace-SR006.nfs2/echerepanov/datasets/data_mikasa_robo" \
#     --ckpt-dir="." \
#     --num-train-data=1000

bash examples/baselines/sac/experiments/1_exp_SWEEP/exp_RollBall.sh #examples/baselines/sac/experiments/1_exp_SWEEP/exp_StackCube.sh #examples/baselines/sac/experiments/1_exp_SWEEP/exp_PokeCube.sh #examples/baselines/sac/experiments/1_exp_SWEEP/exp_PickCube.sh #examples/baselines/sac/experiments/1_exp_SWEEP/exp_PushT.sh #examples/baselines/sac/experiments/1_exp_SWEEP/exp_OpenCabinetDrawer.sh
