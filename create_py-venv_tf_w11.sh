#!/bin/bash
set -ex  # Verbose and stop on error

sudo apt update
sudo apt install -y build-essential curl wget git unzip zip \
    software-properties-common ca-certificates apt-transport-https \
    gnupg lsb-release

sudo apt install -y python3 python3-venv python3-pip

# Create virtual environment with Python 3.10
python3 -m venv DRL-tf-ubuntu

# Activate the virtual environment (Git Bash style)
source DRL-tf-ubuntu/bin/activate

# Upgrade pip using full path (Windows quirk fix)
pip install --upgrade pip

# Install dependencies
pip install -r requirements_venv_tf_w11.txt

# Show what's installed
pip list