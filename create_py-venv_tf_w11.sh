#!/bin/bash
set -ex  # Verbose and stop on error

# Create virtual environment with Python 3.10
py -3.10 -m venv DRL-tf-ubuntu

# Activate the virtual environment (Git Bash style)
source ./DRL-tf-ubuntu/bin/activate

# Upgrade pip using full path (Windows quirk fix)
pip install --upgrade pip

# Install dependencies
#pip install -r requirements_venv_tf_w11.txt

# Show what's installed
pip list