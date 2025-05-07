#!/usr/bin/env bash
# setup.sh — one‑shot installer for Contra‑PPO project on Ubuntu

set -e
set -u
set -o pipefail

######################### 1. 系统依赖 ################################
echo ">>> Installing system packages (build tools, ffmpeg, sdl)…"
sudo apt-get update -qq
sudo apt-get install -y \
    build-essential cmake pkg-config \
    ffmpeg libavcodec-dev libavformat-dev libswscale-dev \
    libsdl2-dev libgl1-mesa-dev libglu1-mesa-dev zlib1g-dev
#################### 2. conda env ###################################
ENV_NAME=contra_ppo
PY_VER=3.11.11

echo ">>> Creating conda env ${ENV_NAME} (Python ${PY_VER}) ..."
eval "$(conda shell.bash hook)"
conda create -y -n ${ENV_NAME} python=${PY_VER}
conda activate ${ENV_NAME}

#################### 3. Python deps #################################
echo ">>> Installing requirements ..."
pip install -r requirements.txt

echo ">>> Installing stable‑retro from local ..."
pip install ./stable-retro
