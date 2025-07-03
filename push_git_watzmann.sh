#!/bin/bash

# Check if commit message was passed
if [ -m "$1" ]; then
  echo "Usage: $0 \"commit message\""
  exit 1
fi

# Start SSH agent
eval "$(ssh-agent -s)"

# Add private key
ssh-add ssh_watzzman_git

# Git operations
git add .
git commit -m "$1"
git remote set-url origin git@github.com:polsuarezm/AFC-DRL-experiment.git
git push origin main