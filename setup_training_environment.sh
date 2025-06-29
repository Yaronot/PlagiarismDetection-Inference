#!/bin/bash
echo "=== GPU Training Environment Setup ==="

# Create virtual environment for training if it doesn't exist
if [ ! -d "/tmp/training_gpu_env" ]; then
    echo "Creating new training virtual environment..."
    python3 -m venv /tmp/training_gpu_env
    
    # Activate and install packages
    source /tmp/training_gpu_env/bin/activate
    
    # Load CUDA
    module load cuda/12.4.1
    
    echo "Installing CUDA PyTorch for training..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers pandas numpy scikit-learn matplotlib seaborn tqdm
    
    echo "Installing additional training dependencies..."
    pip install tensorboard wandb  # For training monitoring (optional)
else
    echo "Using existing training virtual environment..."
    source /tmp/training_gpu_env/bin/activate
    module load cuda/12.4.1
fi

# Set cache directories
export HF_HOME=/tmp/hf_training_cache_$USER
export TRANSFORMERS_CACHE=/tmp/hf_training_cache_$USER
export TORCH_HOME=/tmp/torch_training_cache_$USER

# Create cache and work directories
mkdir -p /tmp/hf_training_cache_$USER /tmp/torch_training_cache_$USER /tmp/training_work
chmod 700 /tmp/hf_training_cache_$USER /tmp/torch_training_cache_$USER

# Test GPU
echo "Testing GPU access for training..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✓ GPU available: {torch.cuda.get_device_name(0)}')
    print(f'✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'✓ CUDA version: {torch.version.cuda}')
    
    # Test GPU computation
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print('✓ GPU computation test successful!')
else:
    print('❌ GPU not available')
    exit(1)
"

# Download BERT model if not cached
if [ ! -d "/tmp/hf_training_cache_$USER/models--bert-base-uncased" ]; then
    echo "Downloading BERT model for training..."
    python3 -c "
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
print('✓ BERT model ready for training')
"
else
    echo "✓ BERT model already cached"
fi

# Go to work directory
cd /tmp/training_work

echo "=== Training Setup Complete! ==="
echo "Current directory: $(pwd)"
echo "Environment: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU memory available: $(python -c 'import torch; print(f\"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")' 2>/dev/null || echo 'N/A')"
