#!/usr/bin/env bash

echo "Initializing environment..."

# Detect OS
OS="$(uname -s)"

# Windows Git Bash returns MINGW* or CYGWIN*
if [[ "$OS" == MINGW* || "$OS" == CYGWIN* ]]; then
    echo "Detected Windows"
    VENV=".venv/Scripts/activate"
else
    echo "Detected Linux or macOS"
    VENV=".venv/bin/activate"
fi

# Pull latest code
echo "Pulling latest changes..."
git pull

# Create venv if missing
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv || python -m venv .venv
else
    echo "Virtual environment already exists."
fi

# Activate venv
echo "Activating virtual environment..."
source "$VENV"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing requirements..."
pip install -r requirements.txt

echo "Environment is ready."
