#!/bin/bash
echo "=== GPU Environment Setup ==="

# Create virtual environment if it doesn't exist
if [ ! -d "/tmp/plagiarism_gpu_env" ]; then
    echo "Creating new virtual environment..."
    python3 -m venv /tmp/plagiarism_gpu_env
    
    # Activate and install packages
    source /tmp/plagiarism_gpu_env/bin/activate
    
    # Load CUDA
    module load cuda/12.4.1
    
    echo "Installing CUDA PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers pandas numpy scikit-learn matplotlib seaborn tqdm
else
    echo "Using existing virtual environment..."
    source /tmp/plagiarism_gpu_env/bin/activate
    module load cuda/12.4.1
fi

# Set cache directories
export HF_HOME=/tmp/hf_cache_$USER
export TRANSFORMERS_CACHE=/tmp/hf_cache_$USER
export TORCH_HOME=/tmp/torch_cache_$USER

# Create cache directories
mkdir -p /tmp/hf_cache_$USER /tmp/torch_cache_$USER /tmp/plagiarism_work
chmod 700 /tmp/hf_cache_$USER /tmp/torch_cache_$USER

# Test GPU
echo "Testing GPU access..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✓ GPU available: {torch.cuda.get_device_name(0)}')
    print(f'✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('❌ GPU not available')
    exit(1)
"

# Download BERT model if not cached
if [ ! -d "/tmp/hf_cache_$USER/models--bert-base-uncased" ]; then
    echo "Downloading BERT model..."
    python3 -c "
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
print('✓ BERT model ready')
"
else
    echo "✓ BERT model already cached"
fi

# Go to work directory
cd /tmp/plagiarism_work

echo "=== Setup complete! ==="
echo "Current directory: $(pwd)"
echo "Environment: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
