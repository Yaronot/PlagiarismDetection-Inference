#!/bin/bash

echo "Creating training session management scripts..."

# ============================================
# TRAINING SESSION START SCRIPT: ~/start_training_session.sh
# ============================================

cat << 'EOF' > ~/start_training_session.sh
#!/bin/bash
echo "=== Starting GPU Plagiarism Model Training Session ==="

# Request GPU node with more resources for training
echo "Requesting GPU node for training (this may take a few minutes)..."
srun --partition=catfish --gres=gpu:l4:1 --cpus-per-task=16 --mem=32G --time=6:00:00 --pty bash -c "

# Load environment
echo 'Setting up GPU training environment...'
source ~/setup_training_environment.sh

# Copy training code to work directory
echo 'Copying training files...'
cp ~/plagiarism_project/suspicious-document/plagiarism_detector.py /tmp/training_work/
cp ~/plagiarism_project/suspicious-document/run_plagiarism_detection.sh /tmp/training_work/
cp ~/plagiarism_project/suspicious-document/all_plagiarism_mappings.csv /tmp/training_work/
chmod +x /tmp/training_work/run_plagiarism_detection.sh

# Copy corpus data
echo 'Setting up corpus data...'
mkdir -p /tmp/training_work/corpus
cp -r ~/plagiarism_project/source-document /tmp/training_work/corpus/
cp -r ~/plagiarism_project/suspicious-document/part* /tmp/training_work/corpus/suspicious-document/

echo '=== Ready! You are now on GPU node with training environment set up ==='
echo 'Available commands:'
echo '  ./run_plagiarism_detection.sh --sample_size 2000 --epochs 5'
echo '  python plagiarism_detector.py --corpus_path ./corpus --csv_file all_plagiarism_mappings.csv'
echo 'REMEMBER: Copy results back with: ~/backup_training_and_exit.sh'

# Start interactive shell
bash
"
EOF

# =============================================
# TRAINING BACKUP SCRIPT: ~/backup_training_and_exit.sh
# =============================================

cat << 'EOF' > ~/backup_training_and_exit.sh
#!/bin/bash
echo "=== Backing up training results before exit ==="

# Copy all training results back to home
if [ -d "/tmp/training_work" ]; then
    echo "Copying training results from /tmp/training_work/ to home..."
    
    # Copy model files
    cp /tmp/training_work/*.pth ~/ 2>/dev/null
    cp /tmp/training_work/*.json ~/ 2>/dev/null
    cp /tmp/training_work/*.png ~/ 2>/dev/null
    cp /tmp/training_work/*.log ~/ 2>/dev/null
    cp /tmp/training_work/*.txt ~/ 2>/dev/null
    
    # Copy to plagiarism_output directory if it exists
    if [ -d "~/plagiarism_project/suspicious-document/plagiarism_output" ]; then
        echo "Also copying to plagiarism_output directory..."
        cp /tmp/training_work/*.pth ~/plagiarism_project/suspicious-document/plagiarism_output/ 2>/dev/null
        cp /tmp/training_work/*.png ~/plagiarism_project/suspicious-document/plagiarism_output/ 2>/dev/null
        cp /tmp/training_work/*.json ~/plagiarism_project/suspicious-document/plagiarism_output/ 2>/dev/null
    fi
    
    echo "Files copied to home directory:"
    ls -la ~/*.pth ~/*.png ~/*.json ~/training_*.log 2>/dev/null
    
    echo "Training artifacts backed up successfully!"
else
    echo "No training work directory found in /tmp/"
fi

echo "=== Training backup complete! Safe to exit. ==="
echo "Next training session: run ~/start_training_session.sh"

# Exit the GPU session
exit
EOF

# =============================================
# TRAINING SETUP SCRIPT: ~/setup_training_environment.sh  
# =============================================

cat << 'EOF' > ~/setup_training_environment.sh
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
EOF

# =============================================
# TRAINING WRAPPER SCRIPT: ~/train_plagiarism_model.sh
# =============================================

cat << 'EOF' > ~/train_plagiarism_model.sh
#!/bin/bash

# Training wrapper script with common configurations
echo "=== Plagiarism Model Training Wrapper ==="

# Parse arguments
SAMPLE_SIZE=2000
EPOCHS=5
BATCH_SIZE=16
LEARNING_RATE=2e-5
OUTPUT_DIR="/tmp/training_work"

while [[ $# -gt 0 ]]; do
    case $1 in
        --sample_size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --sample_size N      Sample size for training (default: 2000)"
            echo "  --epochs N           Number of epochs (default: 5)"
            echo "  --batch_size N       Batch size (default: 16)"
            echo "  --learning_rate F    Learning rate (default: 2e-5)"
            echo "  -h, --help          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Training Configuration:"
echo "  Sample Size: $SAMPLE_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Output Directory: $OUTPUT_DIR"
echo ""

# Update the corpus path to point to our temporary location
CORPUS_PATH="/tmp/training_work/corpus"

# Run training
python3 plagiarism_detector.py \
    --corpus_path "$CORPUS_PATH" \
    --csv_file all_plagiarism_mappings.csv \
    --sample_size $SAMPLE_SIZE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=== Training Complete! ==="
echo "Results saved in: $OUTPUT_DIR"
echo "Don't forget to run: ~/backup_training_and_exit.sh"
EOF

# =============================================
# QUICK TRAINING STATUS SCRIPT: ~/check_training_status.sh
# =============================================

cat << 'EOF' > ~/check_training_status.sh
#!/bin/bash
echo "=== Training Status Check ==="

# Check if training session is running
if pgrep -f "plagiarism_detector.py" > /dev/null; then
    echo "✓ Training process is running"
    echo "Process details:"
    ps aux | grep plagiarism_detector.py | grep -v grep
else
    echo "❌ No training process found"
fi

# Check GPU usage
echo ""
echo "GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
else
    echo "nvidia-smi not available"
fi

# Check training files
echo ""
echo "Training Files:"
if [ -d "/tmp/training_work" ]; then
    echo "Work directory exists:"
    ls -la /tmp/training_work/*.pth /tmp/training_work/*.png /tmp/training_work/*.json 2>/dev/null || echo "No training outputs yet"
else
    echo "No training work directory found"
fi

echo ""
echo "Recent log entries:"
if [ -f "/tmp/training_work/training.log" ]; then
    tail -10 /tmp/training_work/training.log
else
    echo "No training log found"
fi
EOF

# Make all scripts executable
chmod +x ~/start_training_session.sh
chmod +x ~/backup_training_and_exit.sh  
chmod +x ~/setup_training_environment.sh
chmod +x ~/train_plagiarism_model.sh
chmod +x ~/check_training_status.sh

echo "=== All Training Session Management Scripts Created! ==="
echo ""
echo "Created files:"
echo "  ~/start_training_session.sh     - Start new GPU training session"
echo "  ~/backup_training_and_exit.sh   - Save training results and exit"  
echo "  ~/setup_training_environment.sh - Training environment setup"
echo "  ~/train_plagiarism_model.sh     - Training wrapper with common configs"
echo "  ~/check_training_status.sh      - Check training progress"
echo ""
echo "TRAINING WORKFLOW:"
echo "1. Start training session: ~/start_training_session.sh"
echo "2. Run training: ./train_plagiarism_model.sh --sample_size 3000 --epochs 5"
echo "3. Check progress: ~/check_training_status.sh"
echo "4. Before exit: ~/backup_training_and_exit.sh"
echo ""
echo "Alternative manual training:"
echo "  python plagiarism_detector.py --corpus_path ./corpus --csv_file all_plagiarism_mappings.csv --sample_size 2000"
echo ""
echo "Ready to train!"
