#!/bin/bash

# GPU Detection and Setup Script for Cluster
# Usage: ./check_gpu_setup.sh

echo "==================================="
echo "GPU Detection and Setup for Cluster"
echo "==================================="

echo "1. Checking GPU availability..."

# Check for NVIDIA GPUs (CUDA)
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected (CUDA available)"
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv
    echo ""
    echo "CUDA Version:"
    nvidia-smi | grep "CUDA Version"
    GPU_TYPE="cuda"
    
# Check for AMD GPUs (ROCm)
elif command -v rocm-smi &> /dev/null; then
    echo "✓ AMD GPU detected (ROCm available)"
    echo ""
    echo "GPU Information:"
    rocm-smi
    GPU_TYPE="rocm"
    
# Check for Intel GPUs
elif command -v intel_gpu_top &> /dev/null; then
    echo "✓ Intel GPU detected"
    intel_gpu_top -l
    GPU_TYPE="intel"
    
else
    echo "❌ No GPU detected or GPU tools not available"
    echo ""
    echo "Checking if this is a SLURM cluster..."
    
    # Check for SLURM (common in academic clusters)
    if command -v sinfo &> /dev/null; then
        echo "✓ SLURM detected"
        echo ""
        echo "Available GPU partitions:"
        sinfo -o "%P %G %N" | grep -i gpu
        echo ""
        echo "GPU nodes information:"
        scontrol show partition | grep -i gpu -A 5
        GPU_TYPE="slurm"
    else
        echo "❌ No SLURM detected"
        GPU_TYPE="none"
    fi
fi

echo ""
echo "==================================="
echo "2. Python GPU Libraries Check"
echo "==================================="

echo "Checking PyTorch GPU support..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
else:
    print('No CUDA support in PyTorch')
"

echo ""
echo "Checking if MPS (Apple Silicon) is available..."
python3 -c "
import torch
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✓ MPS (Apple Silicon) available')
else:
    print('❌ MPS not available')
"

echo ""
echo "==================================="
echo "3. Cluster Environment Check"
echo "==================================="

echo "Current environment variables:"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-not set}"
echo "SLURM_GPUS: ${SLURM_GPUS:-not set}"
echo "SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE:-not set}"

echo ""
echo "Checking job scheduler..."
if [ -n "$SLURM_JOB_ID" ]; then
    echo "✓ Running in SLURM job: $SLURM_JOB_ID"
    echo "Allocated GPUs: ${SLURM_GPUS_ON_NODE:-0}"
elif [ -n "$PBS_JOBID" ]; then
    echo "✓ Running in PBS job: $PBS_JOBID"
elif [ -n "$LSB_JOBID" ]; then
    echo "✓ Running in LSF job: $LSB_JOBID"
else
    echo "❌ Not running in a job scheduler"
    echo "You may need to submit a GPU job to access GPUs"
fi

echo ""
echo "==================================="
echo "4. Memory and Storage Check"
echo "==================================="

echo "Available RAM:"
free -h

echo ""
echo "Available disk space:"
df -h .

echo ""
echo "==================================="
echo "5. Recommendations"
echo "==================================="

case $GPU_TYPE in
    "cuda")
        echo "✓ CUDA GPUs detected - you can use GPU acceleration!"
        echo ""
        echo "Recommended setup:"
        echo "1. Install CUDA-enabled PyTorch:"
        echo "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        echo ""
        echo "2. Set environment variables:"
        echo "   export CUDA_VISIBLE_DEVICES=0  # Use first GPU"
        echo ""
        echo "3. Verify installation:"
        echo "   python3 -c \"import torch; print(torch.cuda.is_available())\""
        ;;
    "slurm")
        echo "✓ SLURM cluster detected"
        echo ""
        echo "To request GPU resources, use:"
        echo "sbatch --gres=gpu:1 your_job_script.sh"
        echo "or"
        echo "srun --gres=gpu:1 --pty bash"
        echo ""
        echo "Example job script:"
        cat << 'EOF'
#!/bin/bash
#SBATCH --job-name=plagiarism
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G

# Load modules if needed
# module load cuda/11.8
# module load python/3.9

# Activate virtual environment
source plagiarism_env/bin/activate

# Run your analysis
./run_plagiarism_analysis.sh --max_articles 10
EOF
        ;;
    "rocm")
        echo "✓ AMD GPU detected"
        echo ""
        echo "Install ROCm-enabled PyTorch:"
        echo "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2"
        ;;
    *)
        echo "❌ No GPU acceleration available"
        echo ""
        echo "Running on CPU only. Consider:"
        echo "1. Requesting a GPU node if this is a cluster"
        echo "2. Using the optimization strategies for CPU"
        echo "3. Reducing the number of articles for testing"
        ;;
esac

echo ""
echo "==================================="
echo "6. Quick GPU Test"
echo "==================================="

if [ "$GPU_TYPE" = "cuda" ]; then
    echo "Testing GPU with a simple PyTorch operation..."
    python3 -c "
import torch
if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    z = torch.mm(x, y)
    print('✓ GPU computation successful!')
    print(f'Used GPU: {torch.cuda.get_device_name()}')
    print(f'GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
else:
    print('❌ GPU test failed - CUDA not available')
"
fi

echo ""
echo "Setup check complete!"
