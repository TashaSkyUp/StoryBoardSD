#!/bin/bash
#########################################################
# Uncomment and change the variables below to your need:#
#########################################################

# Install directory without trailing slash
#install_dir="/home/$(whoami)"

# Name of the subdirectory
#clone_dir="stable-diffusion-webui"

# Commandline arguments for webui.py, for example: export COMMANDLINE_ARGS="--medvram --opt-split-attention"
export COMMANDLINE_ARGS="--disable-safe-unpickle --xformers --port 7860 --nowebui --server-name 0.0.0.0 --listen"
export STORYBOARD_BENCHMARKS_PATH=/home/ubuntu/StoryBoardSD/SB_BENCH
export STORYBOARD_RENDER_PATH=/home/ubuntu/StoryBoardSD/SB_RENDERS
export STORYBOARD_DEV_MODE=False
export STORYBOARD_PRODUCT=clash
export STORYBOARD_FFMPEG_PATH=/usr/bin/ffmpeg
export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=7996
export STORYBOARD_API_ROLE=sd_server
export STORYBOARD_API_MODEL_PATH=/home/ubuntu/StoryBoardSD/models/Stable-diffusion/v2-1_512-ema-pruned.ckpt

# python3 executable
#python_cmd="python3"

# git executable
#export GIT="git"

# python3 venv without trailing slash (defaults to ${install_dir}/${clone_dir}/venv)
#venv_dir="venv"

# script to launch to start the app
#export LAUNCH_SCRIPT="launch.py"

# install command for torch
#export TORCH_COMMAND="pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113"

# Requirements file to use for stable-diffusion-webui
#export REQS_FILE="requirements_versions.txt"

# Fixed git repos
#export K_DIFFUSION_PACKAGE=""
#export GFPGAN_PACKAGE=""

# Fixed git commits
#export STABLE_DIFFUSION_COMMIT_HASH=""
#export TAMING_TRANSFORMERS_COMMIT_HASH=""
#export CODEFORMER_COMMIT_HASH=""
#export BLIP_COMMIT_HASH=""

# Uncomment to enable accelerated launch
#export ACCELERATE="True"

###########################################
