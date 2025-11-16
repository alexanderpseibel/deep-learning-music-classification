#!/usr/bin/env bash

echo "Initializing environment..."

# Detect OS
OS="$(uname -s)"
echo "Detected OS: $OS"

# Choose correct venv activation path
if [[ "$OS" == MINGW* || "$OS" == CYGWIN* ]]; then
    # Windows Git Bash
    VENV=".venv/Scripts/activate"
    IS_WINDOWS=true
else
    # Linux / macOS / UCloud
    VENV=".venv/bin/activate"
    IS_WINDOWS=false
fi

# Pull latest changes
echo "Pulling latest changes from git..."
git pull

# Create venv if missing
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv || python -m venv .venv
else
    echo "Virtual environment already exists."
fi

# Activate the venv
echo "Activating virtual environment..."
source "$VENV"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing requirements..."
pip install -r requirements.txt

# Install ipykernel ONLY on non-Windows (UCloud)
if [ "$IS_WINDOWS" = false ]; then
    echo "Installing Jupyter kernel for this venv (UCloud only)..."
    pip install ipykernel
    python -m ipykernel install --user --name fma-venv --display-name "Python (fma-venv)"
else
    echo "Skipping Jupyter kernel install on Windows."
fi

echo "Environment setup complete."
