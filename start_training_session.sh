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
