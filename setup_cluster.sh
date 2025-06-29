#!/bin/bash

# Cluster Setup Script for Plagiarism Detection
# Usage: ./setup_cluster.sh

set -e  # Exit on any error

echo "==================================="
echo "Plagiarism Detection Cluster Setup"
echo "==================================="

# Check Python version
echo "Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "Error: Python 3 is required but not found"
    exit 1
fi

# Check if we're in a virtual environment (recommended)
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Using virtual environment: $VIRTUAL_ENV"
else
    echo "Warning: Not in a virtual environment. Consider creating one:"
    echo "  python3 -m venv plagiarism_env"
    echo "  source plagiarism_env/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if pip is available
echo "Checking pip..."
python3 -m pip --version
if [ $? -ne 0 ]; then
    echo "Error: pip is required but not found"
    exit 1
fi

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Check for CUDA availability (for PyTorch)
echo "Checking for CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    CUDA_AVAILABLE=true
else
    echo "CUDA not detected. Will install CPU-only PyTorch."
    CUDA_AVAILABLE=false
fi

# Install PyTorch based on CUDA availability
echo "Installing PyTorch..."
if [ "$CUDA_AVAILABLE" = true ]; then
    # Install CUDA version - check https://pytorch.org for latest
    echo "Installing CUDA-enabled PyTorch..."
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    # Install CPU-only version
    echo "Installing CPU-only PyTorch..."
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "Installing other requirements..."
python3 -m pip install -r requirements.txt

# Verify installation
echo "Verifying installation..."

echo "Testing imports..."
python3 -c "
import torch
import transformers
import pandas
import numpy
import sklearn
import matplotlib
import seaborn
import tqdm
print('✓ All imports successful!')
"

echo "Testing CUDA (if available)..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('Running on CPU')
"

echo "Testing Transformers..."
python3 -c "
from transformers import AutoTokenizer
print('✓ Transformers working!')
"

# Check available space
echo "Checking disk space..."
df -h .

# Download and cache a small model to test everything works
echo "Testing model download (this will download ~400MB)..."
python3 -c "
from transformers import AutoTokenizer, AutoModel
print('Downloading bert-base-uncased for testing...')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
print('✓ Model download and loading successful!')
print(f'Model config: {model.config.hidden_size} hidden size')
"

echo ""
echo "==================================="
echo "Setup completed successfully!"
echo "==================================="
echo ""
echo "Your environment is ready for plagiarism detection."
echo ""
echo "Next steps:"
echo "1. Make sure your model weights are in place:"
echo "   ls -la best_siamese_bert.pth"
echo ""
echo "2. Check if your extract_paragraphs.py script is available:"
echo "   ls -la extract_paragraphs.py"
echo ""
echo "3. Run the plagiarism detection:"
echo "   ./run_plagiarism_analysis.sh --max_articles 2"
echo ""
echo "Useful commands:"
echo "  - Check GPU usage: nvidia-smi"
echo "  - Monitor processes: htop"
echo "  - Check Python packages: python3 -m pip list"
