#!/bin/bash

# Create Virtual Environment for Plagiarism Detection
# Usage: ./create_venv.sh

set -e

echo "==================================="
echo "Creating Virtual Environment"
echo "==================================="

# Check if virtual environment already exists
if [ -d "plagiarism_env" ]; then
    echo "Virtual environment 'plagiarism_env' already exists."
    read -p "Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf plagiarism_env
        echo "Removed existing environment."
    else
        echo "Using existing environment."
        echo "To activate: source plagiarism_env/bin/activate"
        exit 0
    fi
fi

# Create virtual environment
echo "Creating virtual environment 'plagiarism_env'..."
python3 -m venv plagiarism_env

# Activate virtual environment
echo "Activating virtual environment..."
source plagiarism_env/bin/activate

# Upgrade pip in the virtual environment
echo "Upgrading pip in virtual environment..."
python -m pip install --upgrade pip

echo ""
echo "==================================="
echo "Virtual environment created successfully!"
echo "==================================="
echo ""
echo "To activate the environment, run:"
echo "  source plagiarism_env/bin/activate"
echo ""
echo "To deactivate when done, run:"
echo "  deactivate"
echo ""
echo "Next steps:"
echo "1. Activate the environment: source plagiarism_env/bin/activate"
echo "2. Run setup: ./setup_cluster.sh"
