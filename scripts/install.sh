#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=================================================="
echo "Installing UAV Pothole Recognition Environment..."
echo "=================================================="

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Define project root and script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Error: requirements.txt not found."
    exit 1
fi

echo "=================================================="
echo "Environment successfully installed!"
echo "To activate virtual environment: source .venv/bin/activate"
echo "=================================================="
