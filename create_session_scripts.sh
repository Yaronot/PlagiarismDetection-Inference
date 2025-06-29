#!/bin/bash

echo "Creating session management scripts..."

# ============================================
# SESSION START SCRIPT: ~/start_gpu_session.sh
# ============================================

cat << 'EOF' > ~/start_gpu_session.sh
#!/bin/bash
echo "=== Starting GPU Plagiarism Detection Session ==="

# Request GPU node
echo "Requesting GPU node (this may take a few minutes)..."
srun --partition=catfish --gres=gpu:l4:1 --cpus-per-task=8 --mem=16G --time=3:00:00 --pty bash -c "

# Load environment
echo 'Setting up GPU environment...'
source ~/setup_gpu_plagiarism.sh

# Copy code to work directory
echo 'Copying code files...'
cp ~/apply_plagiarism_detection.py /tmp/plagiarism_work/
cp ~/run_plagiarism_analysis.sh /tmp/plagiarism_work/
cp ~/extract_paragraphs.py /tmp/plagiarism_work/
cp ~/best_siamese_bert.pth /tmp/plagiarism_work/
chmod +x /tmp/plagiarism_work/run_plagiarism_analysis.sh

echo '=== Ready! You are now on GPU node with everything set up ==='
echo 'Run: ./run_plagiarism_analysis.sh --max_articles 10'
echo 'REMEMBER: Copy results back with: cp plagiarism_results.* ~/'

# Start interactive shell
bash
"
EOF

# =============================================
# SESSION END SCRIPT: backup_and_exit.sh
# =============================================

cat << 'EOF' > ~/backup_and_exit.sh
#!/bin/bash
echo "=== Backing up work before exit ==="

# Copy all results back to home
if [ -d "/tmp/plagiarism_work" ]; then
    echo "Copying results from /tmp/plagiarism_work/ to home..."
    cp /tmp/plagiarism_work/plagiarism_results.* ~/ 2>/dev/null
    cp /tmp/plagiarism_work/*.txt ~/ 2>/dev/null
    cp /tmp/plagiarism_work/*.log ~/ 2>/dev/null
    cp /tmp/plagiarism_work/*.out ~/ 2>/dev/null
    
    echo "Files copied to home directory:"
    ls -la ~/plagiarism_results.* ~/analysis_*.txt ~/verbose_*.txt 2>/dev/null
else
    echo "No work directory found in /tmp/"
fi

echo "=== Backup complete! Safe to exit. ==="
echo "Next session: run ~/start_gpu_session.sh"

# Exit the GPU session
exit
EOF

# =============================================
# IMPROVED SETUP SCRIPT: ~/setup_gpu_plagiarism.sh  
# =============================================

cat << 'EOF' > ~/setup_gpu_plagiarism.sh
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
EOF

# Make all scripts executable
chmod +x ~/start_gpu_session.sh
chmod +x ~/backup_and_exit.sh  
chmod +x ~/setup_gpu_plagiarism.sh

echo "=== All Session Management Scripts Created! ==="
echo ""
echo "Created files:"
echo "  ~/start_gpu_session.sh    - Start new GPU session"
echo "  ~/backup_and_exit.sh      - Save results and exit"  
echo "  ~/setup_gpu_plagiarism.sh - Environment setup"
echo ""
echo "NEW WORKFLOW:"
echo "1. Start session: ~/start_gpu_session.sh"
echo "2. Run analysis: ./run_plagiarism_analysis.sh --max_articles 10"  
echo "3. Before exit: ~/backup_and_exit.sh"
echo ""
echo "Ready to use!"
